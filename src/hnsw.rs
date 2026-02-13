//! Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search
//!
//! HNSW creates a multi-layer graph structure where:
//! - Layer 0 contains all vectors
//! - Higher layers contain progressively fewer vectors
//! - Each node connects to M nearest neighbors per layer
//! - Search starts at the top layer and descends to layer 0

use crate::distance;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// Distance metric used for nearest-neighbor search
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
}

/// Max-heap element: pop() returns the element with the LARGEST distance.
/// Used for the result set (`nearest`) to evict the farthest neighbor.
#[derive(Clone)]
struct MaxDistElement {
    id: String,
    distance: f32,
}

impl PartialEq for MaxDistElement {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxDistElement {}

impl Ord for MaxDistElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MaxDistElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap element: pop() returns the element with the SMALLEST distance.
/// Used for the candidate queue to explore closest nodes first.
#[derive(Clone)]
struct MinDistElement {
    id: String,
    distance: f32,
}

impl PartialEq for MinDistElement {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MinDistElement {}

impl Ord for MinDistElement {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MinDistElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Node in the HNSW graph
#[derive(Clone, Serialize, Deserialize)]
struct HNSWNode {
    id: String,
    vector: Vec<f32>,
    /// Connections per layer: layer_idx -> set of neighbor IDs
    connections: Vec<HashSet<String>>,
}

/// HNSW Index
#[derive(Clone, Serialize, Deserialize)]
pub struct HNSWIndex {
    pub dimensions: usize,
    /// M: max number of connections per layer
    m: usize,
    /// ef_construction: size of dynamic candidate list during construction
    ef_construction: usize,
    /// Distance metric used for search
    pub metric: DistanceMetric,
    /// All nodes in the index
    nodes: HashMap<String, HNSWNode>,
    /// Entry point (top-level node)
    entry_point: Option<String>,
    /// Maximum layer in the index
    max_layer: usize,
    /// Layer assignment multiplier
    ml: f32,
}

impl HNSWIndex {
    /// Create a new HNSW index
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `m` - Max connections per layer (typically 16-32)
    /// * `ef_construction` - Dynamic list size during construction (typically 200)
    /// * `metric` - Distance metric to use
    pub fn new(dimensions: usize, m: usize, ef_construction: usize, metric: DistanceMetric) -> Self {
        HNSWIndex {
            dimensions,
            m,
            ef_construction,
            metric,
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            ml: 1.0 / (m as f32).ln(),
        }
    }

    /// Check if a vector with the given ID exists
    pub fn contains(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    /// Get the vector data for a given ID
    pub fn get_vector(&self, id: &str) -> Option<&Vec<f32>> {
        self.nodes.get(id).map(|node| &node.vector)
    }

    /// Get all vector IDs
    pub fn all_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Get the number of nodes in the index
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, id: String, vector: Vec<f32>) {
        if vector.len() != self.dimensions {
            return;
        }

        // Determine layer for new node (exponential decay)
        let layer = self.random_layer();

        // Create new node
        let mut node = HNSWNode {
            id: id.clone(),
            vector: vector.clone(),
            connections: vec![HashSet::new(); layer + 1],
        };

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(id.clone());
            self.max_layer = layer;
            self.nodes.insert(id, node);
            return;
        }

        // Find nearest neighbors at each layer
        let entry = self.entry_point.clone().unwrap();
        let mut curr_nearest = vec![entry.clone()];

        // Search from top to target layer
        for lc in (layer + 1..=self.max_layer).rev() {
            let results = self.search_layer(&vector, curr_nearest, 1, lc);
            curr_nearest = results.into_iter().map(|(id, _)| id).collect();
        }

        // Insert and connect at layers 0..=layer
        for lc in (0..=layer).rev() {
            let candidates = self.search_layer(&vector, curr_nearest.clone(), self.ef_construction, lc);
            let candidate_ids: Vec<String> = candidates.into_iter().map(|(id, _)| id).collect();

            // Select M neighbors
            let m = if lc == 0 { self.m * 2 } else { self.m };
            let neighbors = self.select_neighbors(&vector, candidate_ids, m);

            // Add bidirectional connections
            let max_conn = if lc == 0 { self.m * 2 } else { self.m };
            let mut to_prune = Vec::new();

            for neighbor_id in &neighbors {
                node.connections[lc].insert(neighbor_id.clone());

                if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                    // Only add bidirectional connection if neighbor exists at this layer
                    if lc < neighbor.connections.len() {
                        neighbor.connections[lc].insert(id.clone());

                        // Check if pruning needed
                        if neighbor.connections[lc].len() > max_conn {
                            to_prune.push(neighbor_id.clone());
                        }
                    }
                }
            }

            // Prune connections in separate pass
            for neighbor_id in to_prune {
                let pruned = self.prune_connections(&neighbor_id, lc, max_conn);
                if let Some(neighbor) = self.nodes.get_mut(&neighbor_id) {
                    neighbor.connections[lc] = pruned;
                }
            }

            curr_nearest = neighbors.into_iter().collect();
        }

        // Update entry point if new node is at a higher layer
        if layer > self.max_layer {
            self.max_layer = layer;
            self.entry_point = Some(id.clone());
        }

        self.nodes.insert(id, node);
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(String, f32)> {
        if self.entry_point.is_none() {
            return vec![];
        }

        let entry = self.entry_point.clone().unwrap();
        let mut curr_nearest = vec![entry];

        // Search from top to layer 1
        for lc in (1..=self.max_layer).rev() {
            let results = self.search_layer(query, curr_nearest, 1, lc);
            curr_nearest = results.into_iter().map(|(id, _)| id).collect();
        }

        // Search at layer 0
        let candidates = self.search_layer(query, curr_nearest, ef.max(k), 0);

        // Return top k with final distances
        candidates
            .into_iter()
            .take(k)
            .map(|(id, dist)| {
                // For Euclidean, internal computations use squared distance;
                // convert to actual Euclidean distance for the final result
                let final_dist = match self.metric {
                    DistanceMetric::Euclidean => dist.sqrt(),
                    _ => dist,
                };
                (id, final_dist)
            })
            .collect()
    }

    /// Delete a vector by ID
    pub fn delete(&mut self, id: &str) -> bool {
        if !self.nodes.contains_key(id) {
            return false;
        }

        // Remove all connections to this node
        let node = self.nodes.get(id).unwrap().clone();
        for (layer, neighbors) in node.connections.iter().enumerate() {
            for neighbor_id in neighbors {
                if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                    if layer < neighbor.connections.len() {
                        neighbor.connections[layer].remove(id);
                    }
                }
            }
        }

        // Remove the node
        self.nodes.remove(id);

        // Update entry point if needed
        if self.entry_point.as_ref() == Some(&id.to_string()) {
            // Pick the node with the most layers as new entry point
            self.entry_point = self
                .nodes
                .values()
                .max_by_key(|n| n.connections.len())
                .map(|n| n.id.clone());
            self.max_layer = self
                .nodes
                .values()
                .map(|n| n.connections.len().saturating_sub(1))
                .max()
                .unwrap_or(0);
        }

        true
    }

    /// Search within a specific layer
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<String>,
        ef: usize,
        layer: usize,
    ) -> Vec<(String, f32)> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<MinDistElement> = BinaryHeap::new();
        let mut nearest: BinaryHeap<MaxDistElement> = BinaryHeap::new();

        for ep in entry_points {
            let dist = self.distance_to(&ep, query);
            candidates.push(MinDistElement {
                id: ep.clone(),
                distance: dist,
            });
            nearest.push(MaxDistElement {
                id: ep.clone(),
                distance: dist,
            });
            visited.insert(ep);
        }

        while let Some(curr) = candidates.pop() {
            // curr is the closest unexplored candidate
            let furthest_dist = nearest.peek().map(|h| h.distance).unwrap_or(f32::INFINITY);

            if curr.distance > furthest_dist {
                break;
            }

            if let Some(node) = self.nodes.get(&curr.id) {
                if layer < node.connections.len() {
                    for neighbor_id in &node.connections[layer] {
                        if visited.insert(neighbor_id.clone()) {
                            let dist = self.distance_to(neighbor_id, query);
                            let furthest =
                                nearest.peek().map(|h| h.distance).unwrap_or(f32::INFINITY);

                            if dist < furthest || nearest.len() < ef {
                                candidates.push(MinDistElement {
                                    id: neighbor_id.clone(),
                                    distance: dist,
                                });
                                nearest.push(MaxDistElement {
                                    id: neighbor_id.clone(),
                                    distance: dist,
                                });

                                if nearest.len() > ef {
                                    nearest.pop(); // removes the farthest element
                                }
                            }
                        }
                    }
                }
            }
        }

        // into_sorted_vec() returns ascending order = nearest first
        nearest
            .into_sorted_vec()
            .into_iter()
            .map(|h| (h.id, h.distance))
            .collect()
    }

    /// Select best neighbors using heuristic
    fn select_neighbors(&self, _query: &[f32], candidates: Vec<String>, m: usize) -> HashSet<String> {
        candidates.into_iter().take(m).collect()
    }

    /// Prune connections for a node
    fn prune_connections(&self, node_id: &str, layer: usize, max_conn: usize) -> HashSet<String> {
        if let Some(node) = self.nodes.get(node_id) {
            let mut neighbors: Vec<_> = node.connections[layer]
                .iter()
                .map(|id| {
                    let dist = self.distance_between(node_id, id);
                    (id.clone(), dist)
                })
                .collect();

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            neighbors.into_iter().take(max_conn).map(|(id, _)| id).collect()
        } else {
            HashSet::new()
        }
    }

    /// Calculate distance using the configured metric
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => distance::euclidean_distance_squared(a, b),
            DistanceMetric::Cosine => distance::cosine_distance(a, b),
            DistanceMetric::DotProduct => {
                // For dot product, negate so that higher dot product = smaller "distance"
                -distance::dot_product(a, b)
            }
        }
    }

    /// Calculate distance to a query vector
    fn distance_to(&self, id: &str, query: &[f32]) -> f32 {
        self.nodes
            .get(id)
            .map(|node| self.compute_distance(&node.vector, query))
            .unwrap_or(f32::INFINITY)
    }

    /// Calculate distance between two nodes
    fn distance_between(&self, id1: &str, id2: &str) -> f32 {
        match (self.nodes.get(id1), self.nodes.get(id2)) {
            (Some(n1), Some(n2)) => self.compute_distance(&n1.vector, &n2.vector),
            _ => f32::INFINITY,
        }
    }

    /// Random layer assignment (exponential decay)
    fn random_layer(&self) -> usize {
        let mut buf = [0u8; 4];
        getrandom::getrandom(&mut buf).unwrap_or_default();
        let random_val = f32::from_bits(u32::from_le_bytes(buf)).abs() / f32::MAX;
        // Clamp to avoid ln(0) = -inf
        let clamped = random_val.max(f32::MIN_POSITIVE);
        let layer = (-clamped.ln() * self.ml) as usize;
        layer.min(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::DistanceMetric;
    use crate::vector::random_vector_seeded;

    /// Helper: create a deterministic vector from a seed
    fn make_vec(dims: usize, seed: u64) -> Vec<f32> {
        random_vector_seeded(dims, seed)
    }

    // ── Construction & basics ──────────────────────────────────────

    #[test]
    fn new_creates_empty_index() {
        let idx = HNSWIndex::new(128, 16, 200, DistanceMetric::Euclidean);
        assert_eq!(idx.dimensions, 128);
        assert_eq!(idx.m, 16);
        assert_eq!(idx.ef_construction, 200);
        assert_eq!(idx.metric, DistanceMetric::Euclidean);
        assert!(idx.entry_point.is_none());
        assert_eq!(idx.nodes.len(), 0);
    }

    #[test]
    fn first_insert_sets_entry_point() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        assert_eq!(idx.entry_point, Some("a".into()));
        assert_eq!(idx.nodes.len(), 1);
    }

    #[test]
    fn size_tracking_after_insertions() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        for i in 0..10 {
            idx.insert(format!("v{}", i), make_vec(3, i as u64));
        }
        assert_eq!(idx.nodes.len(), 10);
    }

    // ── Insert & search correctness ────────────────────────────────

    #[test]
    fn insert_one_search_finds_it() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        let v = vec![1.0, 0.0, 0.0];
        idx.insert("a".into(), v.clone());
        let results = idx.search(&v, 1, 50);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 1e-6); // distance ~0
    }

    #[test]
    fn insert_two_search_returns_correct_order() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        let close = vec![1.0, 0.0, 0.0];
        let far = vec![10.0, 10.0, 10.0];
        idx.insert("close".into(), close.clone());
        idx.insert("far".into(), far);

        let results = idx.search(&close, 2, 50);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "close");
        assert!(results[0].1 < results[1].1);
    }

    #[test]
    fn search_returns_k_sorted_by_distance() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        for i in 0..15 {
            idx.insert(format!("v{}", i), make_vec(3, i as u64 * 7 + 42));
        }
        let query = make_vec(3, 999);
        let results = idx.search(&query, 5, 50);
        assert_eq!(results.len(), 5);
        // Verify sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn search_k_greater_than_size_returns_all() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]);
        let results = idx.search(&[0.5, 0.5, 0.0], 100, 200);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        let results = idx.search(&[1.0, 0.0, 0.0], 5, 50);
        assert!(results.is_empty());
    }

    // ── Nearest-neighbor quality ───────────────────────────────────

    #[test]
    fn search_finds_true_nearest_neighbor() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        let target = vec![5.0, 5.0, 5.0];
        let nearest = vec![5.1, 5.1, 5.1]; // very close to target
        let far1 = vec![100.0, 0.0, 0.0];
        let far2 = vec![0.0, 100.0, 0.0];

        idx.insert("nearest".into(), nearest);
        idx.insert("far1".into(), far1);
        idx.insert("far2".into(), far2);

        let results = idx.search(&target, 1, 50);
        assert_eq!(results[0].0, "nearest");
    }

    #[test]
    fn cluster_search_finds_cluster_before_outlier() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        // Cluster around origin
        idx.insert("c0".into(), vec![0.1, 0.1, 0.1]);
        idx.insert("c1".into(), vec![0.2, 0.0, 0.1]);
        idx.insert("c2".into(), vec![0.0, 0.2, 0.1]);
        // Outlier
        idx.insert("outlier".into(), vec![50.0, 50.0, 50.0]);

        let results = idx.search(&[0.0, 0.0, 0.0], 4, 50);
        // All cluster members should come before outlier
        let outlier_pos = results.iter().position(|(id, _)| id == "outlier").unwrap();
        assert_eq!(outlier_pos, 3); // outlier is last
    }

    // ── Dimension validation ───────────────────────────────────────

    #[test]
    fn insert_wrong_dimension_is_ignored() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("good".into(), vec![1.0, 0.0, 0.0]);
        idx.insert("bad".into(), vec![1.0, 0.0]); // wrong dimensions
        assert_eq!(idx.nodes.len(), 1);
        assert!(!idx.nodes.contains_key("bad"));
    }

    // ── Delete ─────────────────────────────────────────────────────

    #[test]
    fn delete_existing_returns_true() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        assert!(idx.delete("a"));
        assert_eq!(idx.nodes.len(), 0);
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        assert!(!idx.delete("nope"));
    }

    #[test]
    fn delete_entry_point_search_still_works() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]);
        idx.insert("c".into(), vec![0.0, 0.0, 1.0]);

        let entry = idx.entry_point.clone().unwrap();
        idx.delete(&entry);

        // Search still works with remaining nodes
        let results = idx.search(&[0.5, 0.5, 0.0], 2, 50);
        assert!(!results.is_empty());
        // Deleted entry should not appear
        for (id, _) in &results {
            assert_ne!(id, &entry);
        }
    }

    #[test]
    fn delete_all_vectors_empties_index() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]);
        idx.delete("a");
        idx.delete("b");
        assert_eq!(idx.nodes.len(), 0);
        let results = idx.search(&[1.0, 0.0, 0.0], 5, 50);
        assert!(results.is_empty());
    }

    #[test]
    fn insert_delete_reinsert_same_id() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        idx.delete("a");
        idx.insert("a".into(), vec![0.0, 1.0, 0.0]);
        assert_eq!(idx.nodes.len(), 1);
        let results = idx.search(&[0.0, 1.0, 0.0], 1, 50);
        assert_eq!(results[0].0, "a");
    }

    // ── Serialization round-trip ───────────────────────────────────

    #[test]
    fn serialize_deserialize_preserves_search_results() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]);
        idx.insert("c".into(), vec![0.0, 0.0, 1.0]);

        let query = vec![0.9, 0.1, 0.0];
        let results_before = idx.search(&query, 3, 50);

        let json = serde_json::to_string(&idx).unwrap();
        let idx2: HNSWIndex = serde_json::from_str(&json).unwrap();
        let results_after = idx2.search(&query, 3, 50);

        assert_eq!(results_before.len(), results_after.len());
        for (a, b) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }

    #[test]
    fn serialize_empty_index() {
        let idx = HNSWIndex::new(128, 16, 200, DistanceMetric::Euclidean);
        let json = serde_json::to_string(&idx).unwrap();
        let idx2: HNSWIndex = serde_json::from_str(&json).unwrap();
        assert!(idx2.entry_point.is_none());
        assert_eq!(idx2.nodes.len(), 0);
        assert_eq!(idx2.dimensions, 128);
    }

    #[test]
    fn serialize_after_deletions() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]);
        idx.insert("c".into(), vec![0.0, 0.0, 1.0]);
        idx.delete("b");

        let json = serde_json::to_string(&idx).unwrap();
        let idx2: HNSWIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(idx2.nodes.len(), 2);
        assert!(!idx2.nodes.contains_key("b"));
        // Search still works
        let results = idx2.search(&[1.0, 0.0, 0.0], 2, 50);
        assert_eq!(results.len(), 2);
    }

    // ── Connection integrity ───────────────────────────────────────

    #[test]
    fn connections_are_bidirectional_within_shared_layers() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        for i in 0..10 {
            idx.insert(format!("v{}", i), make_vec(3, i as u64 * 13 + 1));
        }

        // Connections should be bidirectional when both nodes exist on the same layer.
        // A high-layer node may connect to a low-layer node unidirectionally
        // (the low-layer node doesn't have connections at that layer).
        for (id, node) in &idx.nodes {
            for (layer, neighbors) in node.connections.iter().enumerate() {
                for neighbor_id in neighbors {
                    let neighbor = idx.nodes.get(neighbor_id).unwrap();
                    if layer < neighbor.connections.len() {
                        assert!(
                            neighbor.connections[layer].contains(id),
                            "Missing reverse connection: {} -> {} at layer {}",
                            neighbor_id, id, layer
                        );
                    }
                    // If neighbor doesn't have this layer, unidirectional is expected
                }
            }
        }
    }

    #[test]
    fn no_dangling_references_after_delete() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        for i in 0..10 {
            idx.insert(format!("v{}", i), make_vec(3, i as u64 * 7 + 3));
        }
        idx.delete("v5");

        for (_id, node) in &idx.nodes {
            for neighbors in &node.connections {
                assert!(!neighbors.contains("v5"), "Dangling reference to deleted node v5");
            }
        }
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn large_ef_does_not_panic() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        let results = idx.search(&[1.0, 0.0, 0.0], 1, 10000);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn k_zero_returns_empty() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]);
        let results = idx.search(&[1.0, 0.0, 0.0], 0, 50);
        assert!(results.is_empty());
    }

    // ── Distance metric tests ──────────────────────────────────────

    #[test]
    fn cosine_metric_returns_correct_order() {
        let mut idx = HNSWIndex::new(3, 16, 200, DistanceMetric::Cosine);
        // Same direction as query (cosine distance ~ 0)
        idx.insert("same_dir".into(), vec![2.0, 0.0, 0.0]);
        // Orthogonal (cosine distance ~ 1)
        idx.insert("ortho".into(), vec![0.0, 1.0, 0.0]);
        // Opposite (cosine distance ~ 2)
        idx.insert("opposite".into(), vec![-1.0, 0.0, 0.0]);

        let results = idx.search(&[1.0, 0.0, 0.0], 3, 50);
        assert_eq!(results[0].0, "same_dir");
        assert_eq!(results[2].0, "opposite");
    }
}

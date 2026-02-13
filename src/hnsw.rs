/// Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search
///
/// HNSW creates a multi-layer graph structure where:
/// - Layer 0 contains all vectors
/// - Higher layers contain progressively fewer vectors
/// - Each node connects to M nearest neighbors per layer
/// - Search starts at the top layer and descends to layer 0

use crate::distance;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// Priority queue element for search
#[derive(Clone)]
struct HeapElement {
    id: String,
    distance: f32,
}

impl PartialEq for HeapElement {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapElement {}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for HeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
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
    pub fn new(dimensions: usize, m: usize, ef_construction: usize) -> Self {
        HNSWIndex {
            dimensions,
            m,
            ef_construction,
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            ml: 1.0 / (m as f32).ln(),
        }
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
            curr_nearest = self.search_layer(&vector, curr_nearest, 1, lc);
        }

        // Insert and connect at layers 0..=layer
        for lc in (0..=layer).rev() {
            let candidates = self.search_layer(&vector, curr_nearest.clone(), self.ef_construction, lc);

            // Select M neighbors
            let m = if lc == 0 { self.m * 2 } else { self.m };
            let neighbors = self.select_neighbors(&vector, candidates, m);

            // Add bidirectional connections
            let max_conn = if lc == 0 { self.m * 2 } else { self.m };
            let mut to_prune = Vec::new();

            for neighbor_id in &neighbors {
                node.connections[lc].insert(neighbor_id.clone());

                if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                    neighbor.connections[lc].insert(id.clone());

                    // Check if pruning needed
                    if neighbor.connections[lc].len() > max_conn {
                        to_prune.push(neighbor_id.clone());
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
            curr_nearest = self.search_layer(query, curr_nearest, 1, lc);
        }

        // Search at layer 0
        let candidates = self.search_layer(query, curr_nearest, ef.max(k), 0);

        // Return top k
        candidates
            .into_iter()
            .take(k)
            .map(|id| {
                let dist = self.distance_to(&id, query);
                (id, dist)
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
            self.entry_point = self.nodes.keys().next().cloned();
            self.max_layer = self.nodes
                .values()
                .map(|n| n.connections.len() - 1)
                .max()
                .unwrap_or(0);
        }

        true
    }

    /// Search within a specific layer
    fn search_layer(&self, query: &[f32], entry_points: Vec<String>, ef: usize, layer: usize) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut nearest = BinaryHeap::new();

        for ep in entry_points {
            let dist = self.distance_to(&ep, query);
            candidates.push(HeapElement { id: ep.clone(), distance: dist });
            nearest.push(HeapElement { id: ep.clone(), distance: dist });
            visited.insert(ep);
        }

        while let Some(curr) = candidates.pop() {
            let furthest_dist = nearest.peek().map(|h| h.distance).unwrap_or(f32::INFINITY);

            if curr.distance > furthest_dist {
                break;
            }

            if let Some(node) = self.nodes.get(&curr.id) {
                if layer < node.connections.len() {
                    for neighbor_id in &node.connections[layer] {
                        if visited.insert(neighbor_id.clone()) {
                            let dist = self.distance_to(neighbor_id, query);
                            let furthest = nearest.peek().map(|h| h.distance).unwrap_or(f32::INFINITY);

                            if dist < furthest || nearest.len() < ef {
                                candidates.push(HeapElement { id: neighbor_id.clone(), distance: dist });
                                nearest.push(HeapElement { id: neighbor_id.clone(), distance: dist });

                                if nearest.len() > ef {
                                    nearest.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        nearest.into_sorted_vec().into_iter().map(|h| h.id).collect()
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

    /// Calculate distance to a query vector
    fn distance_to(&self, id: &str, query: &[f32]) -> f32 {
        self.nodes
            .get(id)
            .map(|node| distance::euclidean_distance(&node.vector, query))
            .unwrap_or(f32::INFINITY)
    }

    /// Calculate distance between two nodes
    fn distance_between(&self, id1: &str, id2: &str) -> f32 {
        match (self.nodes.get(id1), self.nodes.get(id2)) {
            (Some(n1), Some(n2)) => distance::euclidean_distance(&n1.vector, &n2.vector),
            _ => f32::INFINITY,
        }
    }

    /// Random layer assignment (exponential decay)
    fn random_layer(&self) -> usize {
        // Simple deterministic "random" based on current size
        let random_val = (self.nodes.len() as f32 * 0.123456).fract();
        let layer = (-random_val.ln() * self.ml) as usize;
        layer.min(16) // Cap at 16 layers
    }
}

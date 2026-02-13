mod distance;
mod hnsw;
mod vector;

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vector search result
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<HashMap<String, String>>,
}

/// Vector record for storage
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
}

/// Main VectorDB class - exposed to JavaScript
#[wasm_bindgen]
pub struct VectorDB {
    hnsw_index: hnsw::HNSWIndex,
    metadata: HashMap<String, HashMap<String, String>>,
}

#[wasm_bindgen]
impl VectorDB {
    /// Create a new VectorDB instance
    #[wasm_bindgen(constructor)]
    pub fn new(dimensions: usize, m: usize, ef_construction: usize, metric: Option<String>) -> VectorDB {
        let distance_metric = match metric.as_deref() {
            Some("cosine") => hnsw::DistanceMetric::Cosine,
            Some("dotproduct") | Some("dot_product") => hnsw::DistanceMetric::DotProduct,
            _ => hnsw::DistanceMetric::Euclidean,
        };
        VectorDB {
            hnsw_index: hnsw::HNSWIndex::new(dimensions, m, ef_construction, distance_metric),
            metadata: HashMap::new(),
        }
    }

    /// Insert a vector into the database
    pub fn insert(&mut self, id: String, vector: Vec<f32>, metadata: JsValue) -> Result<(), JsValue> {
        if vector.len() != self.hnsw_index.dimensions {
            return Err(JsValue::from_str(&format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.hnsw_index.dimensions,
                vector.len()
            )));
        }

        // Validate vector values
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(JsValue::from_str("Vector contains NaN or Infinity values"));
        }

        // Parse metadata if provided
        let meta: Option<HashMap<String, String>> = if metadata.is_null() || metadata.is_undefined() {
            None
        } else {
            serde_wasm_bindgen::from_value(metadata).ok()
        };

        // Handle upsert: delete old entry if it exists
        if self.hnsw_index.contains(&id) {
            self.hnsw_index.delete(&id);
        }

        // Add to HNSW index
        self.hnsw_index.insert(id.clone(), vector);

        // Store metadata (replace or remove)
        match meta {
            Some(m) => { self.metadata.insert(id.clone(), m); }
            None => { self.metadata.remove(&id); }
        }

        Ok(())
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: Vec<f32>, k: usize, ef: usize) -> Result<JsValue, JsValue> {
        if query.len() != self.hnsw_index.dimensions {
            return Err(JsValue::from_str(&format!(
                "Query dimension mismatch: expected {}, got {}",
                self.hnsw_index.dimensions,
                query.len()
            )));
        }

        let results = self.hnsw_index.search(&query, k, ef);

        // Manually create JS array to avoid serde_wasm_bindgen HashMap issues
        let js_results = js_sys::Array::new();

        for (id, distance) in results {
            let meta = self.metadata.get(&id);

            let result_obj = js_sys::Object::new();

            // Set id and distance
            js_sys::Reflect::set(&result_obj, &"id".into(), &id.into())?;
            js_sys::Reflect::set(&result_obj, &"distance".into(), &distance.into())?;

            // Manually convert metadata HashMap to JS object
            if let Some(meta_map) = meta {
                let meta_obj = js_sys::Object::new();
                for (key, value) in meta_map {
                    js_sys::Reflect::set(&meta_obj, &key.as_str().into(), &value.as_str().into())?;
                }
                js_sys::Reflect::set(&result_obj, &"metadata".into(), &meta_obj)?;
            } else {
                js_sys::Reflect::set(&result_obj, &"metadata".into(), &JsValue::NULL)?;
            }

            js_results.push(&result_obj);
        }

        Ok(js_results.into())
    }

    /// Get a vector and its metadata by ID
    pub fn get(&self, id: String) -> Result<JsValue, JsValue> {
        match self.hnsw_index.get_vector(&id) {
            Some(vector) => {
                let result_obj = js_sys::Object::new();
                js_sys::Reflect::set(&result_obj, &"id".into(), &id.clone().into())?;

                let js_vec = js_sys::Float32Array::new_with_length(vector.len() as u32);
                js_vec.copy_from(vector);
                js_sys::Reflect::set(&result_obj, &"vector".into(), &js_vec.into())?;

                if let Some(meta_map) = self.metadata.get(&id) {
                    let meta_obj = js_sys::Object::new();
                    for (key, value) in meta_map {
                        js_sys::Reflect::set(&meta_obj, &key.as_str().into(), &value.as_str().into())?;
                    }
                    js_sys::Reflect::set(&result_obj, &"metadata".into(), &meta_obj)?;
                } else {
                    js_sys::Reflect::set(&result_obj, &"metadata".into(), &JsValue::NULL)?;
                }

                Ok(result_obj.into())
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Check if a vector exists by ID
    pub fn has(&self, id: String) -> bool {
        self.hnsw_index.contains(&id)
    }

    /// List all vector IDs
    pub fn list_ids(&self) -> Result<JsValue, JsValue> {
        let ids = self.hnsw_index.all_ids();
        let js_arr = js_sys::Array::new();
        for id in ids {
            js_arr.push(&id.into());
        }
        Ok(js_arr.into())
    }

    /// Delete a vector by ID
    pub fn delete(&mut self, id: String) -> bool {
        self.metadata.remove(&id);
        self.hnsw_index.delete(&id)
    }

    /// Delete multiple vectors by ID, returns number of deletions
    pub fn delete_batch(&mut self, ids: Vec<String>) -> usize {
        let mut count = 0;
        for id in ids {
            self.metadata.remove(&id);
            if self.hnsw_index.delete(&id) {
                count += 1;
            }
        }
        count
    }

    /// Get total number of vectors
    pub fn size(&self) -> usize {
        self.hnsw_index.node_count()
    }

    /// Serialize the entire database to JSON
    pub fn serialize(&self) -> Result<String, JsValue> {
        #[derive(Serialize)]
        struct DBState<'a> {
            version: u32,
            hnsw_index: &'a hnsw::HNSWIndex,
            metadata: &'a HashMap<String, HashMap<String, String>>,
        }

        let state = DBState {
            version: 1,
            hnsw_index: &self.hnsw_index,
            metadata: &self.metadata,
        };

        serde_json::to_string(&state)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deserialize and restore database from JSON
    pub fn deserialize(json: String) -> Result<VectorDB, JsValue> {
        // Try v1 format first
        #[derive(Deserialize)]
        struct DBStateV1 {
            version: u32,
            hnsw_index: hnsw::HNSWIndex,
            metadata: HashMap<String, HashMap<String, String>>,
        }

        // Legacy format (pre-version)
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct DBStateLegacy {
            vectors: HashMap<String, Vec<f32>>,
            metadata: HashMap<String, HashMap<String, String>>,
            hnsw_state: String,
        }

        if let Ok(state) = serde_json::from_str::<DBStateV1>(&json) {
            if state.version != 1 {
                return Err(JsValue::from_str(&format!(
                    "Unsupported database version: {}",
                    state.version
                )));
            }
            return Ok(VectorDB {
                hnsw_index: state.hnsw_index,
                metadata: state.metadata,
            });
        }

        // Fall back to legacy format
        let state: DBStateLegacy = serde_json::from_str(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let hnsw_index: hnsw::HNSWIndex = serde_json::from_str(&state.hnsw_state)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(VectorDB {
            hnsw_index,
            metadata: state.metadata,
        })
    }
}

/// Standalone distance functions exposed to JS
#[wasm_bindgen]
pub fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> Result<f32, JsValue> {
    if a.len() != b.len() {
        return Err(JsValue::from_str("Vectors must have same dimensions"));
    }
    Ok(distance::cosine_similarity(&a, &b))
}

#[wasm_bindgen]
pub fn euclidean_distance(a: Vec<f32>, b: Vec<f32>) -> Result<f32, JsValue> {
    if a.len() != b.len() {
        return Err(JsValue::from_str("Vectors must have same dimensions"));
    }
    Ok(distance::euclidean_distance(&a, &b))
}

#[wasm_bindgen]
pub fn dot_product(a: Vec<f32>, b: Vec<f32>) -> Result<f32, JsValue> {
    if a.len() != b.len() {
        return Err(JsValue::from_str("Vectors must have same dimensions"));
    }
    Ok(distance::dot_product(&a, &b))
}

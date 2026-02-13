mod distance;
mod hnsw;
mod vector;

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Vector search result
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
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
    vectors: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, HashMap<String, String>>,
}

#[wasm_bindgen]
impl VectorDB {
    /// Create a new VectorDB instance
    #[wasm_bindgen(constructor)]
    pub fn new(dimensions: usize, m: usize, ef_construction: usize) -> VectorDB {
        // console_log!("Creating VectorDB with dimensions={}, M={}, ef_construction={}", dimensions, m, ef_construction);
        VectorDB {
            hnsw_index: hnsw::HNSWIndex::new(dimensions, m, ef_construction),
            vectors: HashMap::new(),
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

        // Parse metadata if provided
        let meta: Option<HashMap<String, String>> = if metadata.is_null() || metadata.is_undefined() {
            None
        } else {
            serde_wasm_bindgen::from_value(metadata).ok()
        };

        // console_log!("🔍 Rust insert {}: has_metadata={}, text_len={}",
        //     id,
        //     meta.is_some(),
        //     meta.as_ref().and_then(|m| m.get("text")).map(|t| t.len()).unwrap_or(0)
        // );

        // Add to HNSW index
        self.hnsw_index.insert(id.clone(), vector.clone());

        // Store vector and metadata
        self.vectors.insert(id.clone(), vector);
        if let Some(m) = meta {
            self.metadata.insert(id.clone(), m);
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

        console_log!("🔍 Rust search: found {} results, metadata entries: {}", results.len(), self.metadata.len());

        // Manually create JS array to avoid serde_wasm_bindgen HashMap issues
        let js_results = js_sys::Array::new();

        for (id, score) in results {
            let meta = self.metadata.get(&id);
            console_log!("🔍 Rust result {}: has_metadata={}, text_len={}",
                id,
                meta.is_some(),
                meta.and_then(|m| m.get("text")).map(|t| t.len()).unwrap_or(0)
            );

            let result_obj = js_sys::Object::new();

            // Set id and score
            js_sys::Reflect::set(&result_obj, &"id".into(), &id.into())?;
            js_sys::Reflect::set(&result_obj, &"score".into(), &score.into())?;

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

    /// Delete a vector by ID
    pub fn delete(&mut self, id: String) -> bool {
        self.vectors.remove(&id);
        self.metadata.remove(&id);
        self.hnsw_index.delete(&id)
    }

    /// Get total number of vectors
    pub fn size(&self) -> usize {
        self.vectors.len()
    }

    /// Serialize the entire database to JSON
    pub fn serialize(&self) -> Result<String, JsValue> {
        #[derive(Serialize)]
        struct DBState {
            vectors: HashMap<String, Vec<f32>>,
            metadata: HashMap<String, HashMap<String, String>>,
            hnsw_state: String,
        }

        let state = DBState {
            vectors: self.vectors.clone(),
            metadata: self.metadata.clone(),
            hnsw_state: serde_json::to_string(&self.hnsw_index)
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        };

        serde_json::to_string(&state)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deserialize and restore database from JSON
    pub fn deserialize(json: String) -> Result<VectorDB, JsValue> {
        #[derive(Deserialize)]
        struct DBState {
            vectors: HashMap<String, Vec<f32>>,
            metadata: HashMap<String, HashMap<String, String>>,
            hnsw_state: String,
        }

        let state: DBState = serde_json::from_str(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let hnsw_index: hnsw::HNSWIndex = serde_json::from_str(&state.hnsw_state)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(VectorDB {
            hnsw_index,
            vectors: state.vectors,
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

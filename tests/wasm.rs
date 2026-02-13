//! wasm_bindgen_test tests for the JS-facing VectorDB API
//!
//! Run with: wasm-pack test --headless --chrome
//! Or:       wasm-pack test --node

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use idbvec::*;

// ── VectorDB construction ──────────────────────────────────────

#[wasm_bindgen_test]
fn new_vectordb_has_size_zero() {
    let db = VectorDB::new(3, 16, 200);
    assert_eq!(db.size(), 0);
}

// ── Insert ─────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn insert_increases_size() {
    let mut db = VectorDB::new(3, 16, 200);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    assert_eq!(db.size(), 1);
}

#[wasm_bindgen_test]
fn insert_dimension_mismatch_returns_err() {
    let mut db = VectorDB::new(3, 16, 200);
    let result = db.insert("a".into(), vec![1.0, 0.0], JsValue::NULL);
    assert!(result.is_err());
    assert_eq!(db.size(), 0);
}

#[wasm_bindgen_test]
fn insert_multiple_vectors() {
    let mut db = VectorDB::new(3, 16, 200);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("b".into(), vec![0.0, 1.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("c".into(), vec![0.0, 0.0, 1.0], JsValue::NULL)
        .unwrap();
    assert_eq!(db.size(), 3);
}

// ── Search ─────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn search_returns_results() {
    let mut db = VectorDB::new(3, 16, 200);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("b".into(), vec![0.0, 1.0, 0.0], JsValue::NULL)
        .unwrap();

    let results = db.search(vec![1.0, 0.0, 0.0], 2, 50).unwrap();
    // Results should be a JsValue (array)
    assert!(results.is_object());
}

#[wasm_bindgen_test]
fn search_dimension_mismatch_returns_err() {
    let mut db = VectorDB::new(3, 16, 200);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    let result = db.search(vec![1.0, 0.0], 1, 50);
    assert!(result.is_err());
}

// ── Delete ─────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn delete_existing_returns_true() {
    let mut db = VectorDB::new(3, 16, 200);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    assert!(db.delete("a".into()));
    assert_eq!(db.size(), 0);
}

#[wasm_bindgen_test]
fn delete_nonexistent_returns_false() {
    let mut db = VectorDB::new(3, 16, 200);
    assert!(!db.delete("nope".into()));
}

// ── Serialize / Deserialize ────────────────────────────────────

#[wasm_bindgen_test]
fn serialize_deserialize_roundtrip() {
    let mut db = VectorDB::new(3, 16, 200);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("b".into(), vec![0.0, 1.0, 0.0], JsValue::NULL)
        .unwrap();

    let json = db.serialize().unwrap();
    let db2 = VectorDB::deserialize(json).unwrap();
    assert_eq!(db2.size(), 2);

    // Search still works after deserialization
    let results = db2.search(vec![1.0, 0.0, 0.0], 2, 50).unwrap();
    assert!(results.is_object());
}

#[wasm_bindgen_test]
fn serialize_empty_db() {
    let db = VectorDB::new(5, 16, 200);
    let json = db.serialize().unwrap();
    let db2 = VectorDB::deserialize(json).unwrap();
    assert_eq!(db2.size(), 0);
}

// ── Standalone distance functions ──────────────────────────────

#[wasm_bindgen_test]
fn cosine_similarity_basic() {
    let result = cosine_similarity(vec![1.0, 0.0], vec![1.0, 0.0]).unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}

#[wasm_bindgen_test]
fn cosine_similarity_dimension_mismatch() {
    let result = cosine_similarity(vec![1.0, 0.0], vec![1.0, 0.0, 0.0]);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn euclidean_distance_basic() {
    let result = euclidean_distance(vec![0.0, 0.0], vec![3.0, 4.0]).unwrap();
    assert!((result - 5.0).abs() < 1e-6);
}

#[wasm_bindgen_test]
fn euclidean_distance_dimension_mismatch() {
    let result = euclidean_distance(vec![1.0], vec![1.0, 2.0]);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn dot_product_basic() {
    let result = dot_product(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).unwrap();
    assert!((result - 32.0).abs() < 1e-6);
}

#[wasm_bindgen_test]
fn dot_product_dimension_mismatch() {
    let result = dot_product(vec![1.0, 2.0], vec![3.0]);
    assert!(result.is_err());
}

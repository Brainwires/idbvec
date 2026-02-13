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
    let db = VectorDB::new(3, 16, 200, None);
    assert_eq!(db.size(), 0);
}

#[wasm_bindgen_test]
fn new_vectordb_with_metric() {
    let db = VectorDB::new(3, 16, 200, Some("cosine".into()));
    assert_eq!(db.size(), 0);
}

// ── Insert ─────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn insert_increases_size() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    assert_eq!(db.size(), 1);
}

#[wasm_bindgen_test]
fn insert_dimension_mismatch_returns_err() {
    let mut db = VectorDB::new(3, 16, 200, None);
    let result = db.insert("a".into(), vec![1.0, 0.0], JsValue::NULL);
    assert!(result.is_err());
    assert_eq!(db.size(), 0);
}

#[wasm_bindgen_test]
fn insert_nan_returns_err() {
    let mut db = VectorDB::new(3, 16, 200, None);
    let result = db.insert("a".into(), vec![1.0, f32::NAN, 0.0], JsValue::NULL);
    assert!(result.is_err());
    assert_eq!(db.size(), 0);
}

#[wasm_bindgen_test]
fn insert_infinity_returns_err() {
    let mut db = VectorDB::new(3, 16, 200, None);
    let result = db.insert("a".into(), vec![1.0, f32::INFINITY, 0.0], JsValue::NULL);
    assert!(result.is_err());
    assert_eq!(db.size(), 0);
}

#[wasm_bindgen_test]
fn insert_multiple_vectors() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("b".into(), vec![0.0, 1.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("c".into(), vec![0.0, 0.0, 1.0], JsValue::NULL)
        .unwrap();
    assert_eq!(db.size(), 3);
}

#[wasm_bindgen_test]
fn insert_duplicate_id_upserts() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("a".into(), vec![0.0, 1.0, 0.0], JsValue::NULL)
        .unwrap();
    assert_eq!(db.size(), 1);
}

// ── Search ─────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn search_returns_results() {
    let mut db = VectorDB::new(3, 16, 200, None);
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
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    let result = db.search(vec![1.0, 0.0], 1, 50);
    assert!(result.is_err());
}

// ── Get ───────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn get_existing_returns_object() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    let result = db.get("a".into()).unwrap();
    assert!(result.is_object());
}

#[wasm_bindgen_test]
fn get_nonexistent_returns_null() {
    let db = VectorDB::new(3, 16, 200, None);
    let result = db.get("nope".into()).unwrap();
    assert!(result.is_null());
}

// ── Has ───────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn has_existing_returns_true() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    assert!(db.has("a".into()));
}

#[wasm_bindgen_test]
fn has_nonexistent_returns_false() {
    let db = VectorDB::new(3, 16, 200, None);
    assert!(!db.has("nope".into()));
}

// ── List IDs ──────────────────────────────────────────────────

#[wasm_bindgen_test]
fn list_ids_returns_array() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("b".into(), vec![0.0, 1.0, 0.0], JsValue::NULL)
        .unwrap();
    let ids = db.list_ids().unwrap();
    assert!(ids.is_object());
}

// ── Delete ─────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn delete_existing_returns_true() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    assert!(db.delete("a".into()));
    assert_eq!(db.size(), 0);
}

#[wasm_bindgen_test]
fn delete_nonexistent_returns_false() {
    let mut db = VectorDB::new(3, 16, 200, None);
    assert!(!db.delete("nope".into()));
}

// ── Delete Batch ──────────────────────────────────────────────

#[wasm_bindgen_test]
fn delete_batch_removes_multiple() {
    let mut db = VectorDB::new(3, 16, 200, None);
    db.insert("a".into(), vec![1.0, 0.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("b".into(), vec![0.0, 1.0, 0.0], JsValue::NULL)
        .unwrap();
    db.insert("c".into(), vec![0.0, 0.0, 1.0], JsValue::NULL)
        .unwrap();
    let count = db.delete_batch(vec!["a".into(), "c".into()]);
    assert_eq!(count, 2);
    assert_eq!(db.size(), 1);
}

// ── Serialize / Deserialize ────────────────────────────────────

#[wasm_bindgen_test]
fn serialize_deserialize_roundtrip() {
    let mut db = VectorDB::new(3, 16, 200, None);
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
    let db = VectorDB::new(5, 16, 200, None);
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

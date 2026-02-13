/// Vector data structures and utilities

use serde::{Deserialize, Serialize};
use std::fmt;

/// A vector with an ID and optional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub id: String,
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(id: String, data: Vec<f32>) -> Self {
        Vector { id, data }
    }

    pub fn dimensions(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector(id={}, dims={})", self.id, self.dimensions())
    }
}

/// Helper to create random vectors for testing
#[cfg(test)]
pub fn random_vector(dimensions: usize) -> Vec<f32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let hasher = RandomState::new().build_hasher();
    let seed = hasher.finish();

    let mut rng = seed;
    (0..dimensions)
        .map(|_| {
            // Simple LCG random number generator
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng / 65536) % 32768) as f32 / 32768.0
        })
        .collect()
}

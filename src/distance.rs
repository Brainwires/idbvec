//! Distance and similarity metrics for vectors
//! Optimized for performance with potential SIMD support

/// Compute cosine similarity between two vectors
/// Returns value in range [-1, 1], where 1 means identical direction
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = magnitude(a);
    let norm_b = magnitude(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute cosine distance (1 - cosine_similarity)
/// Returns value in range [0, 2], where 0 means identical vectors
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Compute Euclidean (L2) distance between two vectors
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

/// Compute squared Euclidean distance (avoids sqrt for performance)
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Compute dot product of two vectors
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Compute Manhattan (L1) distance
#[inline]
#[allow(dead_code)]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// Compute vector magnitude (L2 norm)
#[inline]
pub fn magnitude(v: &[f32]) -> f32 {
    v.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
}

/// Normalize a vector to unit length (in-place)
#[allow(dead_code)]
pub fn normalize(v: &mut [f32]) {
    let mag = magnitude(v);
    if mag > 0.0 {
        for x in v.iter_mut() {
            *x /= mag;
        }
    }
}

/// Create a normalized copy of a vector
#[allow(dead_code)]
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let mut result = v.to_vec();
    normalize(&mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── cosine_similarity ──────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_both_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_high_dimensional() {
        // Two identical high-dimensional vectors should have similarity 1.0
        let a: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let b = a.clone();
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_nearly_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0001];
        assert!(cosine_similarity(&a, &b) > 0.999);
    }

    // ── cosine_distance ────────────────────────────────────────────

    #[test]
    fn test_cosine_distance_relationship() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let sim = cosine_similarity(&a, &b);
        let dist = cosine_distance(&a, &b);
        assert!((dist - (1.0 - sim)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 0.0];
        assert!((cosine_distance(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    // ── euclidean_distance ─────────────────────────────────────────

    #[test]
    fn test_euclidean_distance_3_4_5() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((euclidean_distance(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_1d() {
        let a = vec![3.0];
        let b = vec![7.0];
        assert!((euclidean_distance(&a, &b) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_high_dimensional() {
        // All ones vs all zeros in 100D: sqrt(100) = 10
        let a = vec![0.0; 100];
        let b = vec![1.0; 100];
        assert!((euclidean_distance(&a, &b) - 10.0).abs() < 1e-4);
    }

    // ── euclidean_distance_squared ─────────────────────────────────

    #[test]
    fn test_euclidean_distance_squared_relationship() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        let dist = euclidean_distance(&a, &b);
        let dist_sq = euclidean_distance_squared(&a, &b);
        assert!((dist_sq - dist * dist).abs() < 1e-4);
    }

    #[test]
    fn test_euclidean_distance_squared_identical() {
        let a = vec![5.0, 5.0];
        assert!((euclidean_distance_squared(&a, &a) - 0.0).abs() < 1e-6);
    }

    // ── manhattan_distance ─────────────────────────────────────────

    #[test]
    fn test_manhattan_distance_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert!((manhattan_distance(&a, &b) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_axis_aligned() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 0.0];
        assert!((manhattan_distance(&a, &b) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_identical() {
        let a = vec![1.0, 2.0];
        assert!((manhattan_distance(&a, &a) - 0.0).abs() < 1e-6);
    }

    // ── magnitude ──────────────────────────────────────────────────

    #[test]
    fn test_magnitude_unit_vectors() {
        assert!((magnitude(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((magnitude(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((magnitude(&[0.0, 0.0, 1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude_zero_vector() {
        assert_eq!(magnitude(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_magnitude_3_4() {
        assert!((magnitude(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }

    // ── normalize ──────────────────────────────────────────────────

    #[test]
    fn test_normalize_basic() {
        let mut v = vec![3.0, 4.0, 0.0];
        normalize(&mut v);
        assert!((magnitude(&v) - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector_noop() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_already_normalized() {
        let mut v = vec![0.6, 0.8, 0.0];
        normalize(&mut v);
        assert!((magnitude(&v) - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    // ── normalized ─────────────────────────────────────────────────

    #[test]
    fn test_normalized_returns_copy() {
        let v = vec![3.0, 4.0];
        let n = normalized(&v);
        // Original unchanged
        assert_eq!(v, vec![3.0, 4.0]);
        // Copy is normalized
        assert!((magnitude(&n) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_zero_vector() {
        let v = vec![0.0, 0.0];
        let n = normalized(&v);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    // ── dot_product ────────────────────────────────────────────────

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((dot_product(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_self_equals_magnitude_squared() {
        let a = vec![3.0, 4.0];
        let mag = magnitude(&a);
        assert!((dot_product(&a, &a) - mag * mag).abs() < 1e-4);
    }
}

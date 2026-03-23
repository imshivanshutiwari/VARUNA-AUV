//! Softmax and argmax utilities.

/// Compute softmax of a slice.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Return index and value of maximum element.
pub fn argmax(probs: &[f32]) -> (usize, f32) {
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum = {}", sum);
    }

    #[test]
    fn test_argmax_correct() {
        let probs = vec![0.1f32, 0.6, 0.2, 0.1];
        let (idx, val) = argmax(&probs);
        assert_eq!(idx, 1);
        assert!((val - 0.6).abs() < 1e-6);
    }
}

//! ONNX-based acoustic target classifier using tract.
use crate::softmax::{softmax, argmax};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::path::Path;
use tract_onnx::prelude::*;

/// Vessel / target class names.
pub const CLASS_NAMES: &[&str] = &[
    "Cargo",
    "Tanker",
    "Tug",
    "Passengership",
    "Submarine",
    "Biological",
    "Unknown",
];

#[derive(Debug, Error)]
pub enum ClassifierError {
    #[error("model file not found: {0}")]
    ModelNotFound(String),
    #[error("tract inference error: {0}")]
    InferenceError(String),
    #[error("invalid input shape")]
    InvalidInputShape,
}

/// Configuration for the acoustic classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub model_path: String,
    pub input_shape: [usize; 3],
    pub confidence_threshold: f32,
    pub submarine_alert_threshold: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            model_path: "models/acoustic_classifier.onnx".to_string(),
            input_shape: [1, 120, 128],
            confidence_threshold: 0.6,
            submarine_alert_threshold: 0.8,
        }
    }
}

/// A single classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub class_index: usize,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub is_submarine_alert: bool,
    pub above_threshold: bool,
}

impl Default for ClassificationResult {
    fn default() -> Self {
        Self {
            label: "Unknown".to_string(),
            class_index: 6,
            confidence: 0.0,
            probabilities: vec![0.0; 7],
            is_submarine_alert: false,
            above_threshold: false,
        }
    }
}

/// Acoustic target classifier backed by an ONNX model (via tract).
pub struct AcousticClassifier {
    config: ClassifierConfig,
    model: Option<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>,
    classes: Vec<String>,
}

impl AcousticClassifier {
    /// Load model from the configured path. Returns Ok(classifier) on success.
    /// If the model file does not exist, the classifier runs in stub mode.
    pub fn new(config: ClassifierConfig) -> Result<Self, ClassifierError> {
        let classes: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();
        if !Path::new(&config.model_path).exists() {
            tracing::warn!(
                "Model not found at '{}', running in stub mode.",
                config.model_path
            );
            return Ok(Self { config, model: None, classes });
        }
        let model = tract_onnx::onnx()
            .model_for_path(&config.model_path)
            .map_err(|e| ClassifierError::InferenceError(e.to_string()))?
            .into_optimized()
            .map_err(|e| ClassifierError::InferenceError(e.to_string()))?
            .into_runnable()
            .map_err(|e| ClassifierError::InferenceError(e.to_string()))?;

        Ok(Self { config, model: Some(model), classes })
    }

    /// Classify a feature tensor of shape (n_frames, n_mels) or (batch, n_frames, n_mels).
    /// `features` is a flat f32 slice of length n_frames × n_mels.
    pub fn classify(&self, features: &[f32]) -> Result<ClassificationResult, ClassifierError> {
        let logits = if let Some(ref model) = self.model {
            let [_, n_frames, n_mels] = self.config.input_shape;
            let expected = n_frames * n_mels;
            if features.len() < expected {
                return Err(ClassifierError::InvalidInputShape);
            }
            let array = tract_ndarray::Array3::from_shape_vec(
                (1usize, n_frames, n_mels),
                features[..expected].to_vec(),
            )
            .map_err(|_| ClassifierError::InvalidInputShape)?;
            let input: Tensor = array.into_tensor();
            let result = model
                .run(tvec![input.into()])
                .map_err(|e| ClassifierError::InferenceError(e.to_string()))?;
            let out = result[0]
                .to_array_view::<f32>()
                .map_err(|e| ClassifierError::InferenceError(e.to_string()))?;
            out.iter().cloned().collect::<Vec<f32>>()
        } else {
            // Stub mode: deterministic output based on feature energy
            self.stub_classify(features)
        };

        let probs = softmax(&logits);
        let (class_index, confidence) = argmax(&probs);
        let label = self.classes.get(class_index).cloned().unwrap_or_else(|| "Unknown".to_string());
        let is_submarine_alert = class_index == 4 && confidence >= self.config.submarine_alert_threshold;

        Ok(ClassificationResult {
            label,
            class_index,
            confidence,
            probabilities: probs,
            is_submarine_alert,
            above_threshold: confidence >= self.config.confidence_threshold,
        })
    }

    /// Stub classifier: produces deterministic pseudo-probabilities from feature energy.
    fn stub_classify(&self, features: &[f32]) -> Vec<f32> {
        let n_classes = self.classes.len();
        let energy: f32 = features.iter().map(|&x| x * x).sum::<f32>() / features.len().max(1) as f32;
        // Use energy to select a likely class deterministically
        let seed = ((energy * 1000.0) as u64) % n_classes as u64;
        let mut logits = vec![-2.0f32; n_classes];
        logits[seed as usize] = 2.0;
        logits
    }

    pub fn class_names(&self) -> &[String] {
        &self.classes
    }

    pub fn is_stub_mode(&self) -> bool {
        self.model.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_classifier_returns_result() {
        let config = ClassifierConfig::default();
        // Model file won't exist → stub mode
        let clf = AcousticClassifier::new(config).unwrap();
        assert!(clf.is_stub_mode());
        let features = vec![0.5f32; 120 * 128];
        let result = clf.classify(&features).unwrap();
        assert!(!result.label.is_empty());
        assert!(result.confidence > 0.0);
        let prob_sum: f32 = result.probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-4, "Probs sum = {}", prob_sum);
    }

    #[test]
    fn test_class_names_count() {
        let config = ClassifierConfig::default();
        let clf = AcousticClassifier::new(config).unwrap();
        assert_eq!(clf.class_names().len(), CLASS_NAMES.len());
    }
}

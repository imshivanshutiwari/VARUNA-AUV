//! varuna-inference – ONNX model loading and acoustic classification.
pub mod classifier;
pub mod softmax;

pub use classifier::{AcousticClassifier, ClassificationResult, ClassifierConfig};

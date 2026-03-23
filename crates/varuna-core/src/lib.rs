//! varuna-core – DSP, signal processing, beamforming, tracking, and fingerprinting.
pub mod fft;
pub mod signal;
pub mod tracker;
pub mod fingerprint;

pub use fft::stft::Stft;
pub use fft::welch::WelchPsd;
pub use signal::lofar::Lofargram;
pub use signal::demon::DemonProcessor;
pub use signal::mfcc::MfccExtractor;
pub use signal::beamformer::MvdrBeamformer;
pub use signal::cfar::CfarDetector;
pub use tracker::kalman::KalmanTracker;
pub use fingerprint::hasher::AcousticFingerprinter;

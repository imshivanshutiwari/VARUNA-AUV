//! Shared application state.
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use varuna_core::{CfarDetector, DemonProcessor, Lofargram, MfccExtractor, MvdrBeamformer, Stft};
use varuna_inference::{AcousticClassifier, ClassifierConfig};

/// Live sonar / tracking data broadcast to WebSocket clients.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SonarFrame {
    pub timestamp_ms: u64,
    pub spectrum_db: Vec<f32>,
    pub lofar_slice: Vec<f32>,
    pub demon_power_db: Vec<f32>,
    pub demon_blade_rate_hz: f32,
    pub mfcc_frame: Vec<f32>,
    pub doa_deg: f32,
    pub cfar_detections: Vec<CfarDetectionDto>,
    pub classification: ClassificationDto,
    pub tracks: Vec<TrackDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CfarDetectionDto {
    pub frequency_hz: f32,
    pub magnitude_db: f32,
    pub snr_db: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClassificationDto {
    pub label: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub is_submarine_alert: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrackDto {
    pub id: u32,
    pub bearing_deg: f32,
    pub range_m: f32,
    pub label: String,
    pub confidence: f32,
}

#[derive(Clone)]
pub struct AppState {
    pub sonar_frame: Arc<RwLock<SonarFrame>>,
}

impl AppState {
    pub async fn new() -> anyhow::Result<Self> {
        let config = ClassifierConfig::default();
        let classifier = Arc::new(AcousticClassifier::new(config)?);

        // Seed initial sonar frame with synthetic data
        let frame = Self::generate_demo_frame(&classifier);
        let sonar_frame = Arc::new(RwLock::new(frame));

        // Spawn background task to continuously generate synthetic sonar data
        {
            let sonar_frame = sonar_frame.clone();
            let clf = classifier.clone();
            tokio::spawn(async move {
                let mut tick = 0u64;
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    let frame = Self::generate_frame_at(tick, &clf);
                    *sonar_frame.write().await = frame;
                    tick += 1;
                }
            });
        }

        Ok(Self { sonar_frame })
    }

    fn generate_demo_frame(clf: &AcousticClassifier) -> SonarFrame {
        Self::generate_frame_at(0, clf)
    }

    fn generate_frame_at(tick: u64, clf: &AcousticClassifier) -> SonarFrame {
        use std::f32::consts::PI;
        let sr = 44100.0f32;
        let n = 4096usize;
        let t = tick as f32 * 0.05;

        // Synthetic acoustic signal: multiple sinusoids + noise
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let ti = i as f32 / sr;
                0.5 * (2.0 * PI * 100.0 * ti).sin()
                    + 0.3 * (2.0 * PI * 250.0 * (ti + t * 0.1)).sin()
                    + 0.2 * (2.0 * PI * 500.0 * ti).sin()
                    + 0.1 * ((tick * 1234 + i as u64) % 256) as f32 / 128.0
                    - 0.1
            })
            .collect();

        // STFT / spectrum
        let stft = Stft::new(1024, 256);
        let spec = stft.magnitude_db(&signal).unwrap_or_default();
        let spectrum_db: Vec<f32> = if spec.nrows() > 0 {
            spec.row(0).iter().cloned().collect()
        } else {
            vec![-60.0; 513]
        };

        // Welch PSD → spectrum_db already represents that
        // LOFAR
        let lofar = Lofargram::new(4096, 512, sr, 1.0, 500.0, 0.5);
        let lofar_grid = lofar.compute(&signal);
        let lofar_slice: Vec<f32> = if lofar_grid.nrows() > 0 {
            lofar_grid.row(0).iter().cloned().collect()
        } else {
            vec![-100.0; 64]
        };

        // DEMON
        let demon = DemonProcessor::new(2000.0, 8000.0, 100.0, sr);
        let demon_result = demon.compute(&signal);

        // MFCC
        let mfcc_ext = MfccExtractor::new(40, 128, sr, 20.0, 8000.0, 512, 128);
        let mfcc_mat = mfcc_ext.compute(&signal);
        let mfcc_frame: Vec<f32> = if mfcc_mat.nrows() > 0 {
            mfcc_mat.row(0).iter().cloned().collect()
        } else {
            vec![0.0; 40]
        };

        // Beamformer
        let bf = MvdrBeamformer::new(8, 0.037, 1500.0, 0.001, 1.0);
        let snapshots: Vec<Vec<f32>> = (0..32)
            .map(|s| {
                let theta = 30.0f32 * PI / 180.0 + (t * 0.01).sin() * 0.1;
                (0..8)
                    .map(|i| {
                        let tau = i as f32 * 0.037 * theta.sin() / 1500.0;
                        (2.0 * PI * 1000.0 * (tau + s as f32 / 1000.0)).cos()
                    })
                    .collect()
            })
            .collect();
        let beam_pattern = bf.compute(&snapshots);

        // CFAR
        let cfar = CfarDetector::new(2, 8, 1e-4);
        let freq_axis: Vec<f32> = (0..spectrum_db.len())
            .map(|k| k as f32 * sr / 2048.0)
            .collect();
        let cfar_dets = cfar.detect(&spectrum_db, &freq_axis);

        // Classification (use MFCC as feature)
        let features: Vec<f32> = {
            let pad = 120 * 128;
            let mut f = mfcc_frame.clone();
            f.resize(pad, 0.0);
            f
        };
        let clf_result = clf.classify(&features).unwrap_or_default();

        let timestamp_ms = tick * 50;

        SonarFrame {
            timestamp_ms,
            spectrum_db: spectrum_db.clone(),
            lofar_slice,
            demon_power_db: demon_result.power_db.clone(),
            demon_blade_rate_hz: demon_result.blade_rate_hz,
            mfcc_frame,
            doa_deg: beam_pattern.doa_deg,
            cfar_detections: cfar_dets
                .iter()
                .map(|d| CfarDetectionDto {
                    frequency_hz: d.frequency_hz,
                    magnitude_db: d.magnitude_db,
                    snr_db: d.snr_db,
                })
                .collect(),
            classification: ClassificationDto {
                label: clf_result.label,
                confidence: clf_result.confidence,
                probabilities: clf_result.probabilities,
                is_submarine_alert: clf_result.is_submarine_alert,
            },
            tracks: vec![TrackDto {
                id: 1,
                bearing_deg: beam_pattern.doa_deg,
                range_m: 1500.0 + (t * 0.5).sin() * 100.0,
                label: "Cargo".to_string(),
                confidence: 0.82,
            }],
        }
    }
}

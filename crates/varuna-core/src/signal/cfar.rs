//! Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector.

/// A detection result from CFAR.
#[derive(Debug, Clone)]
pub struct CfarDetection {
    /// Index in the input spectrum.
    pub bin_index: usize,
    /// Frequency in Hz.
    pub frequency_hz: f32,
    /// Signal magnitude in dB.
    pub magnitude_db: f32,
    /// Adaptive threshold in dB.
    pub threshold_db: f32,
    /// Signal-to-noise ratio above threshold in dB.
    pub snr_db: f32,
}

/// CA-CFAR (Cell-Averaging Constant False Alarm Rate) detector.
pub struct CfarDetector {
    /// Number of guard cells on each side.
    pub guard_cells: usize,
    /// Number of training cells on each side.
    pub training_cells: usize,
    /// Probability of false alarm (used to compute threshold multiplier).
    pub pfa: f32,
}

impl CfarDetector {
    pub fn new(guard_cells: usize, training_cells: usize, pfa: f32) -> Self {
        Self {
            guard_cells,
            training_cells,
            pfa,
        }
    }

    /// CFAR threshold multiplier α = N * (PFA^{-1/N} - 1) where N = 2*training_cells.
    fn threshold_multiplier(&self) -> f32 {
        let n = 2 * self.training_cells;
        (n as f32) * (self.pfa.powf(-(1.0 / n as f32)) - 1.0)
    }

    /// Detect peaks in `spectrum_db` using CA-CFAR.
    /// `freq_axis_hz`: frequency corresponding to each bin.
    pub fn detect(&self, spectrum_db: &[f32], freq_axis_hz: &[f32]) -> Vec<CfarDetection> {
        let n = spectrum_db.len();
        let window = self.guard_cells + self.training_cells;
        let alpha = self.threshold_multiplier();
        let mut detections = Vec::new();

        for i in window..n.saturating_sub(window) {
            // Gather training cells (skip guard cells)
            let left: Vec<f32> = ((i - window)..(i - self.guard_cells))
                .map(|k| spectrum_db[k])
                .collect();
            let right: Vec<f32> = ((i + self.guard_cells + 1)..=(i + window))
                .filter(|&k| k < n)
                .map(|k| spectrum_db[k])
                .collect();

            let all_training: Vec<f32> = left.into_iter().chain(right).collect();
            if all_training.is_empty() {
                continue;
            }
            let noise_est: f32 = all_training.iter().sum::<f32>() / all_training.len() as f32;
            let threshold = noise_est + 10.0 * (alpha + 1.0).log10();
            if spectrum_db[i] > threshold {
                let freq = if i < freq_axis_hz.len() {
                    freq_axis_hz[i]
                } else {
                    i as f32
                };
                detections.push(CfarDetection {
                    bin_index: i,
                    frequency_hz: freq,
                    magnitude_db: spectrum_db[i],
                    threshold_db: threshold,
                    snr_db: spectrum_db[i] - threshold,
                });
            }
        }
        detections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfar_detects_spike() {
        // Flat noise floor with a spike
        let n = 256usize;
        let mut spectrum = vec![-60.0f32; n];
        spectrum[128] = 0.0; // Strong spike at bin 128
        let freqs: Vec<f32> = (0..n).map(|i| i as f32 * 43.0).collect(); // 43 Hz/bin
        let cfar = CfarDetector::new(2, 8, 1e-4);
        let dets = cfar.detect(&spectrum, &freqs);
        assert!(!dets.is_empty(), "Should detect spike at bin 128");
        assert!(dets.iter().any(|d| d.bin_index == 128));
    }

    #[test]
    fn test_cfar_no_false_alarm_on_flat() {
        let n = 256usize;
        let spectrum = vec![-60.0f32; n];
        let freqs: Vec<f32> = (0..n).map(|i| i as f32 * 43.0).collect();
        let cfar = CfarDetector::new(2, 8, 1e-4);
        let dets = cfar.detect(&spectrum, &freqs);
        assert!(dets.is_empty(), "No detections on flat spectrum");
    }
}

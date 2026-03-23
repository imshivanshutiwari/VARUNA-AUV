//! Short-Time Fourier Transform (STFT) using realfft.
use std::f32::consts::PI;
use ndarray::Array2;
use realfft::RealFftPlanner;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StftError {
    #[error("signal too short for window size {window_size}")]
    SignalTooShort { window_size: usize },
}

/// STFT processor producing magnitude spectrograms.
pub struct Stft {
    pub window_size: usize,
    pub hop_size: usize,
    window: Vec<f32>,
}

impl Stft {
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        let window = hann_window(window_size);
        Self { window_size, hop_size, window }
    }

    /// Number of real-FFT output bins.
    pub fn num_bins(&self) -> usize {
        self.window_size / 2 + 1
    }

    /// Compute magnitude spectrogram (dB scale).
    /// Returns Array2 of shape (n_frames, n_bins).
    pub fn magnitude_db(&self, signal: &[f32]) -> Result<Array2<f32>, StftError> {
        if signal.len() < self.window_size {
            return Err(StftError::SignalTooShort { window_size: self.window_size });
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.window_size);
        let n_frames = (signal.len() - self.window_size) / self.hop_size + 1;
        let n_bins = self.num_bins();
        let mut out = Array2::<f32>::zeros((n_frames, n_bins));

        for (frame_idx, row) in out.rows_mut().into_iter().enumerate() {
            let start = frame_idx * self.hop_size;
            let mut frame: Vec<f32> = signal[start..start + self.window_size]
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut frame, &mut spectrum)
                .expect("FFT failed");
            for (j, c) in spectrum.iter().enumerate() {
                let mag = c.norm();
                row[j] = if mag > 1e-10 { 20.0 * mag.log10() } else { -100.0 };
            }
        }
        Ok(out)
    }

    /// Compute power spectrogram (linear).
    pub fn power(&self, signal: &[f32]) -> Result<Array2<f32>, StftError> {
        if signal.len() < self.window_size {
            return Err(StftError::SignalTooShort { window_size: self.window_size });
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.window_size);
        let n_frames = (signal.len() - self.window_size) / self.hop_size + 1;
        let n_bins = self.num_bins();
        let mut out = Array2::<f32>::zeros((n_frames, n_bins));

        for (frame_idx, row) in out.rows_mut().into_iter().enumerate() {
            let start = frame_idx * self.hop_size;
            let mut frame: Vec<f32> = signal[start..start + self.window_size]
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut frame, &mut spectrum)
                .expect("FFT failed");
            for (j, c) in spectrum.iter().enumerate() {
                row[j] = c.norm_sqr();
            }
        }
        Ok(out)
    }
}

/// Generate a Hann window of the given size.
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (size - 1) as f32).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_output_shape() {
        let window = 256usize;
        let hop = 64usize;
        let n = 2048usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let stft = Stft::new(window, hop);
        let spec = stft.magnitude_db(&signal).unwrap();
        let expected_frames = (n - window) / hop + 1;
        assert_eq!(spec.nrows(), expected_frames);
        assert_eq!(spec.ncols(), window / 2 + 1);
    }

    #[test]
    fn test_stft_dc_component() {
        // DC signal → energy concentrated at bin 0
        let n = 1024usize;
        let signal = vec![1.0f32; n];
        let stft = Stft::new(256, 64);
        let spec = stft.power(&signal).unwrap();
        // Bin 0 should be largest
        let row = spec.row(0);
        let max_bin = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(max_bin, 0);
    }
}

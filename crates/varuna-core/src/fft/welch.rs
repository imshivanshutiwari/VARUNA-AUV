//! Welch's method for Power Spectral Density estimation.
use super::stft::hann_window;
use ndarray::Array1;
use realfft::RealFftPlanner;

/// Welch PSD estimator.
pub struct WelchPsd {
    pub window_size: usize,
    pub hop_size: usize,
    pub sample_rate: f32,
}

impl WelchPsd {
    pub fn new(window_size: usize, hop_size: usize, sample_rate: f32) -> Self {
        Self {
            window_size,
            hop_size,
            sample_rate,
        }
    }

    /// Compute Welch PSD.
    /// Returns (frequencies_hz: Array1<f32>, psd_db: Array1<f32>).
    pub fn compute(&self, signal: &[f32]) -> (Array1<f32>, Array1<f32>) {
        let n = signal.len();
        if n < self.window_size {
            let bins = self.window_size / 2 + 1;
            return (Array1::zeros(bins), Array1::zeros(bins));
        }
        let window = hann_window(self.window_size);
        let window_power: f32 = window.iter().map(|&w| w * w).sum();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.window_size);
        let n_bins = self.window_size / 2 + 1;
        let mut psd_accum = vec![0.0f32; n_bins];
        let mut n_frames = 0usize;
        let mut start = 0;
        while start + self.window_size <= n {
            let mut frame: Vec<f32> = signal[start..start + self.window_size]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut frame, &mut spectrum).expect("FFT");
            for (k, c) in spectrum.iter().enumerate() {
                psd_accum[k] += c.norm_sqr();
            }
            n_frames += 1;
            start += self.hop_size;
        }
        if n_frames == 0 {
            return (Array1::zeros(n_bins), Array1::zeros(n_bins));
        }
        let scale = 1.0 / (n_frames as f32 * window_power * self.sample_rate);
        let freqs: Array1<f32> = Array1::linspace(0.0, self.sample_rate / 2.0, n_bins);
        let psd: Array1<f32> = psd_accum
            .iter()
            .map(|&p| {
                let linear = p * scale;
                if linear > 1e-20 {
                    10.0 * linear.log10()
                } else {
                    -200.0
                }
            })
            .collect();
        (freqs, psd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_welch_pure_tone_peak() {
        let sr = 44100.0f32;
        let freq = 1000.0f32;
        let n = 44100usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect();
        let welch = WelchPsd::new(1024, 512, sr);
        let (freqs, psd) = welch.compute(&signal);
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - freq).abs() < 100.0,
            "Peak at {}Hz, expected ~{}Hz",
            peak_freq,
            freq
        );
    }
}

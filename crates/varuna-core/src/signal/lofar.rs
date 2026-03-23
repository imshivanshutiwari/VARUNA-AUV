//! LOFARgram – Low-Frequency Analysis and Recording spectral display.
use crate::fft::stft::hann_window;
use ndarray::Array2;
use realfft::RealFftPlanner;

/// LOFARgram processor: narrow-band low-frequency spectrogram.
pub struct Lofargram {
    pub window_size: usize,
    pub hop_size: usize,
    pub sample_rate: f32,
    pub freq_min_hz: f32,
    pub freq_max_hz: f32,
    pub integration_frames: usize,
}

impl Lofargram {
    pub fn new(
        window_size: usize,
        hop_size: usize,
        sample_rate: f32,
        freq_min_hz: f32,
        freq_max_hz: f32,
        integration_time_s: f32,
    ) -> Self {
        let frame_duration_s = hop_size as f32 / sample_rate;
        let integration_frames = (integration_time_s / frame_duration_s).max(1.0) as usize;
        Self {
            window_size,
            hop_size,
            sample_rate,
            freq_min_hz,
            freq_max_hz,
            integration_frames,
        }
    }

    /// Compute LOFARgram: returns (n_slices, n_freq_bins) power matrix in dB.
    pub fn compute(&self, signal: &[f32]) -> Array2<f32> {
        if signal.len() < self.window_size {
            return Array2::zeros((0, 1));
        }
        let window = hann_window(self.window_size);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.window_size);
        let n_bins_total = self.window_size / 2 + 1;

        let bin_low = ((self.freq_min_hz / self.sample_rate) * self.window_size as f32) as usize;
        let bin_high = (((self.freq_max_hz / self.sample_rate) * self.window_size as f32) as usize)
            .min(n_bins_total - 1);
        let n_out_bins = if bin_high > bin_low {
            bin_high - bin_low
        } else {
            1
        };

        // Collect all FFT frames
        let mut all_power: Vec<Vec<f32>> = Vec::new();
        let mut start = 0;
        while start + self.window_size <= signal.len() {
            let mut frame: Vec<f32> = signal[start..start + self.window_size]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut frame, &mut spectrum).expect("FFT");
            let power: Vec<f32> = spectrum[bin_low..=bin_high.min(spectrum.len() - 1)]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();
            all_power.push(power);
            start += self.hop_size;
        }
        if all_power.is_empty() {
            return Array2::zeros((0, n_out_bins));
        }

        // Integrate (average) over integration_frames
        let n_slices = (all_power.len() + self.integration_frames - 1) / self.integration_frames;
        let mut out = Array2::<f32>::zeros((n_slices, n_out_bins));
        for (slice_idx, chunk) in all_power.chunks(self.integration_frames).enumerate() {
            for bin in 0..n_out_bins {
                let sum: f32 = chunk.iter().filter_map(|f| f.get(bin)).sum();
                let avg = sum / chunk.len() as f32;
                out[[slice_idx, bin]] = if avg > 1e-20 {
                    10.0 * avg.log10()
                } else {
                    -200.0
                };
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_lofar_output_shape() {
        let sr = 44100.0f32;
        let signal: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 100.0 * i as f32 / sr).sin())
            .collect();
        let lofar = Lofargram::new(4096, 1024, sr, 1.0, 500.0, 0.5);
        let out = lofar.compute(&signal);
        assert!(out.nrows() > 0, "Expected some slices");
        assert!(out.ncols() > 0, "Expected some freq bins");
    }
}

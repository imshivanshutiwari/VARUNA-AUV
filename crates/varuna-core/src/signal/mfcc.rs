//! Mel-Frequency Cepstral Coefficients (MFCC) extractor.
use crate::fft::stft::hann_window;
use ndarray::Array2;
use realfft::RealFftPlanner;

/// MFCC extractor.
pub struct MfccExtractor {
    pub n_mfcc: usize,
    pub n_mels: usize,
    pub sample_rate: f32,
    pub fmin: f32,
    pub fmax: f32,
    pub window_size: usize,
    pub hop_size: usize,
    mel_filterbank: Vec<Vec<f32>>,
}

impl MfccExtractor {
    pub fn new(
        n_mfcc: usize,
        n_mels: usize,
        sample_rate: f32,
        fmin: f32,
        fmax: f32,
        window_size: usize,
        hop_size: usize,
    ) -> Self {
        let n_bins = window_size / 2 + 1;
        let mel_filterbank = build_mel_filterbank(n_mels, n_bins, sample_rate, fmin, fmax);
        Self {
            n_mfcc,
            n_mels,
            sample_rate,
            fmin,
            fmax,
            window_size,
            hop_size,
            mel_filterbank,
        }
    }

    /// Extract MFCC feature matrix of shape (n_frames, n_mfcc).
    pub fn compute(&self, signal: &[f32]) -> Array2<f32> {
        if signal.len() < self.window_size {
            return Array2::zeros((0, self.n_mfcc));
        }
        let window = hann_window(self.window_size);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.window_size);
        let n_frames = (signal.len() - self.window_size) / self.hop_size + 1;
        let mut out = Array2::<f32>::zeros((n_frames, self.n_mfcc));

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_size;
            let mut frame: Vec<f32> = signal[start..start + self.window_size]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut frame, &mut spectrum).expect("FFT");

            // Power spectrum
            let power: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();

            // Apply mel filterbank → log energy
            let mel_energy: Vec<f32> = self
                .mel_filterbank
                .iter()
                .map(|filt| {
                    let e: f32 = filt.iter().zip(power.iter()).map(|(&f, &p)| f * p).sum();
                    if e > 1e-20 {
                        e.ln()
                    } else {
                        -46.0
                    }
                })
                .collect();

            // DCT-II to get MFCCs
            let mfcc = dct2(&mel_energy, self.n_mfcc);
            for (k, &val) in mfcc.iter().enumerate() {
                out[[frame_idx, k]] = val;
            }
        }
        out
    }
}

/// Build mel filterbank: returns Vec of n_mels filters, each of length n_bins.
fn build_mel_filterbank(
    n_mels: usize,
    n_bins: usize,
    sample_rate: f32,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let freq_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = freq_points
        .iter()
        .map(|&f| (f / (sample_rate / 2.0)) * n_bins as f32)
        .collect();

    (0..n_mels)
        .map(|m| {
            let start = bin_points[m];
            let center = bin_points[m + 1];
            let end = bin_points[m + 2];
            (0..n_bins)
                .map(|k| {
                    let kf = k as f32;
                    if kf >= start && kf <= center {
                        (kf - start) / (center - start + 1e-10)
                    } else if kf > center && kf <= end {
                        (end - kf) / (end - center + 1e-10)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// DCT-II of input, returning first n_out coefficients.
fn dct2(input: &[f32], n_out: usize) -> Vec<f32> {
    let n = input.len();
    let pi = std::f32::consts::PI;
    (0..n_out.min(n))
        .map(|k| {
            let scale = if k == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
            scale
                * input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * (pi * k as f32 * (2 * i + 1) as f32 / (2 * n) as f32).cos())
                    .sum::<f32>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_mfcc_output_shape() {
        let sr = 22050.0f32;
        let n = 22050usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();
        let extractor = MfccExtractor::new(40, 128, sr, 20.0, 8000.0, 512, 128);
        let mfcc = extractor.compute(&signal);
        assert_eq!(mfcc.ncols(), 40);
        assert!(mfcc.nrows() > 0);
    }

    #[test]
    fn test_mfcc_not_all_zero() {
        let sr = 22050.0f32;
        let n = 22050usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();
        let extractor = MfccExtractor::new(13, 64, sr, 20.0, 8000.0, 512, 128);
        let mfcc = extractor.compute(&signal);
        let sum: f32 = mfcc.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "MFCCs should not be all zero");
    }
}

//! DEMON (Detection of Envelope Modulations ON Noise) processor.
//! Demodulates propeller cavitation noise to extract blade-rate frequency.
use crate::fft::stft::hann_window;
use realfft::RealFftPlanner;

/// DEMON spectrum result.
#[derive(Debug, Clone)]
pub struct DemonSpectrum {
    /// Frequency axis in Hz (0 to sample_rate/2 of the envelope signal).
    pub frequencies_hz: Vec<f32>,
    /// Power spectrum magnitude in dB.
    pub power_db: Vec<f32>,
    /// Estimated blade-rate frequency in Hz (peak of DEMON spectrum).
    pub blade_rate_hz: f32,
}

/// DEMON processor for extracting propeller signatures.
pub struct DemonProcessor {
    pub band_low_hz: f32,
    pub band_high_hz: f32,
    pub envelope_lpf_hz: f32,
    pub sample_rate: f32,
    pub demon_window_size: usize,
}

impl DemonProcessor {
    pub fn new(
        band_low_hz: f32,
        band_high_hz: f32,
        envelope_lpf_hz: f32,
        sample_rate: f32,
    ) -> Self {
        let demon_window_size = 2048usize;
        Self {
            band_low_hz,
            band_high_hz,
            envelope_lpf_hz,
            sample_rate,
            demon_window_size,
        }
    }

    /// Compute DEMON spectrum from a raw acoustic signal.
    pub fn compute(&self, signal: &[f32]) -> DemonSpectrum {
        // 1. Bandpass filter the signal into the cavitation band
        let filtered = bandpass_approx(
            signal,
            self.band_low_hz,
            self.band_high_hz,
            self.sample_rate,
        );
        // 2. Compute envelope (full-wave rectification)
        let envelope: Vec<f32> = filtered.iter().map(|&x| x.abs()).collect();
        // 3. Lowpass filter the envelope
        let lp_envelope = lowpass_approx(&envelope, self.envelope_lpf_hz, self.sample_rate);
        // 4. Compute FFT of the envelope
        let n = self.demon_window_size.min(lp_envelope.len());
        if n < 4 {
            return DemonSpectrum {
                frequencies_hz: vec![],
                power_db: vec![],
                blade_rate_hz: 0.0,
            };
        }
        let window = hann_window(n);
        let mut frame: Vec<f32> = lp_envelope[..n]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut frame, &mut spectrum).expect("FFT");

        let n_bins = n / 2 + 1;
        // Effective sample rate of the envelope signal ≈ 2×envelope_lpf_hz
        let demon_sr = 2.0 * self.envelope_lpf_hz;
        let freqs: Vec<f32> = (0..n_bins)
            .map(|k| k as f32 * demon_sr / n as f32)
            .collect();
        let power_db: Vec<f32> = spectrum
            .iter()
            .take(n_bins)
            .map(|c| {
                let p = c.norm_sqr();
                if p > 1e-20 {
                    10.0 * p.log10()
                } else {
                    -200.0
                }
            })
            .collect();

        // Skip DC bin (index 0) when finding blade rate
        let blade_rate_hz = if power_db.len() > 1 {
            let peak_idx = power_db[1..]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i + 1)
                .unwrap_or(0);
            freqs[peak_idx]
        } else {
            0.0
        };

        DemonSpectrum {
            frequencies_hz: freqs,
            power_db,
            blade_rate_hz,
        }
    }
}

/// Simple moving-average bandpass approximation.
fn bandpass_approx(signal: &[f32], low: f32, high: f32, sr: f32) -> Vec<f32> {
    // Simple difference of lowpass filters as bandpass approximation
    let lp_high = lowpass_approx(signal, high, sr);
    let lp_low = lowpass_approx(signal, low, sr);
    lp_high
        .iter()
        .zip(lp_low.iter())
        .map(|(&h, &l)| h - l)
        .collect()
}

/// Exponential moving-average lowpass filter.
fn lowpass_approx(signal: &[f32], cutoff_hz: f32, sr: f32) -> Vec<f32> {
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let dt = 1.0 / sr;
    let alpha = dt / (rc + dt);
    let mut out = Vec::with_capacity(signal.len());
    let mut prev = 0.0f32;
    for &s in signal {
        prev = prev + alpha * (s - prev);
        out.push(prev);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_demon_returns_spectrum() {
        let sr = 44100.0f32;
        let n = 22050usize;
        // Simulate AM signal: cavitation carrier modulated at blade rate ~10 Hz
        let blade_rate = 10.0f32;
        let carrier = 5000.0f32;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (1.0 + 0.5 * (2.0 * PI * blade_rate * t).sin()) * (2.0 * PI * carrier * t).sin()
            })
            .collect();
        let demon = DemonProcessor::new(2000.0, 10000.0, 100.0, sr);
        let result = demon.compute(&signal);
        assert!(!result.frequencies_hz.is_empty());
        assert!(!result.power_db.is_empty());
    }
}

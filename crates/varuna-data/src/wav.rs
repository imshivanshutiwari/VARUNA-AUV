//! WAV file reading with automatic mono downmix and resampling stubs.
use hound::{WavSpec, SampleFormat};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WavError {
    #[error("hound I/O error: {0}")]
    Hound(#[from] hound::Error),
    #[error("unsupported sample format")]
    UnsupportedFormat,
    #[error("file not found: {0}")]
    NotFound(String),
}

/// Mono-channel audio buffer with sample rate metadata.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Interleaved mono samples in the range [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Duration in seconds.
    pub duration_s: f32,
}

impl AudioBuffer {
    /// Create from raw f32 samples.
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Self {
        let duration_s = samples.len() as f32 / sample_rate as f32;
        Self { sample_rate, samples, duration_s }
    }
}

/// WAV file reader that normalises to mono f32.
pub struct WavReader;

impl WavReader {
    /// Read a WAV file, returning a mono f32 AudioBuffer.
    pub fn read(path: &str) -> Result<AudioBuffer, WavError> {
        if !std::path::Path::new(path).exists() {
            return Err(WavError::NotFound(path.to_string()));
        }
        let mut reader = hound::WavReader::open(path)?;
        let spec: WavSpec = reader.spec();
        let n_channels = spec.channels as usize;

        let samples: Vec<f32> = match spec.sample_format {
            SampleFormat::Float => {
                reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
            }
            SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .map(|&s| s as f32 / max_val)
                    .collect()
            }
        };

        // Downmix to mono by averaging channels
        let mono: Vec<f32> = if n_channels == 1 {
            samples
        } else {
            samples
                .chunks(n_channels)
                .map(|ch| ch.iter().sum::<f32>() / n_channels as f32)
                .collect()
        };

        Ok(AudioBuffer::new(spec.sample_rate, mono))
    }

    /// Write a mono f32 buffer to a WAV file.
    pub fn write(path: &str, buf: &AudioBuffer) -> Result<(), WavError> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: buf.sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for &s in &buf.samples {
            writer.write_sample(s)?;
        }
        writer.finalize()?;
        Ok(())
    }

    /// Generate a synthetic sinusoidal AudioBuffer for testing.
    pub fn synthetic_sine(
        freq_hz: f32,
        duration_s: f32,
        sample_rate: u32,
    ) -> AudioBuffer {
        let n = (duration_s * sample_rate as f32) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32)
                    .sin()
            })
            .collect();
        AudioBuffer::new(sample_rate, samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_sine_duration() {
        let buf = WavReader::synthetic_sine(440.0, 1.0, 44100);
        assert_eq!(buf.samples.len(), 44100);
        assert!((buf.duration_s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_synthetic_sine_amplitude() {
        let buf = WavReader::synthetic_sine(440.0, 1.0, 44100);
        let max_amp = buf.samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_amp = buf.samples.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max_amp <= 1.0 + 1e-5);
        assert!(min_amp >= -1.0 - 1e-5);
    }
}

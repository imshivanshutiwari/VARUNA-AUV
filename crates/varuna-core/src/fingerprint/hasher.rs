//! Acoustic fingerprinting using spectrogram peak hashing (Shazam-style).
use std::collections::HashMap;
use ndarray::Array2;

/// A single acoustic fingerprint hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FingerprintHash {
    /// Frequency bin of anchor peak.
    pub freq1: u32,
    /// Frequency bin of paired peak.
    pub freq2: u32,
    /// Time delta between anchor and paired peak (in frames).
    pub delta_frames: u32,
}

/// A match between a query and a stored fingerprint.
#[derive(Debug, Clone)]
pub struct FingerprintMatch {
    pub track_id: String,
    pub score: f32,
    pub time_offset_frames: i32,
}

/// Acoustic fingerprinter using spectrogram constellation peak hashing.
pub struct AcousticFingerprinter {
    /// Stored fingerprints: hash → Vec<(track_id, anchor_time)>
    db: HashMap<FingerprintHash, Vec<(String, u32)>>,
    /// Parameters
    pub peak_threshold_db: f32,
    pub fan_out: usize,
    pub time_window_frames: u32,
}

impl AcousticFingerprinter {
    pub fn new(peak_threshold_db: f32, fan_out: usize, time_window_frames: u32) -> Self {
        Self {
            db: HashMap::new(),
            peak_threshold_db,
            fan_out,
            time_window_frames,
        }
    }

    /// Extract constellation peaks from a magnitude spectrogram (n_frames, n_bins).
    fn extract_peaks(&self, spectrogram: &Array2<f32>) -> Vec<(u32, u32)> {
        let n_frames = spectrogram.nrows();
        let n_bins = spectrogram.ncols();
        let mut peaks = Vec::new();
        for t in 1..n_frames.saturating_sub(1) {
            for f in 1..n_bins.saturating_sub(1) {
                let val = spectrogram[[t, f]];
                if val < self.peak_threshold_db { continue; }
                // Local maximum in 3x3 neighbourhood
                let is_peak = (-1i32..=1).flat_map(|dt| (-1i32..=1).map(move |df| (dt, df)))
                    .filter(|&(dt, df)| dt != 0 || df != 0)
                    .all(|(dt, df)| {
                        let tt = (t as i32 + dt) as usize;
                        let ff = (f as i32 + df) as usize;
                        spectrogram[[tt, ff]] <= val
                    });
                if is_peak {
                    peaks.push((t as u32, f as u32));
                }
            }
        }
        peaks
    }

    /// Generate hashes from peaks using the fan-out pairing strategy.
    fn generate_hashes(&self, peaks: &[(u32, u32)]) -> Vec<(FingerprintHash, u32)> {
        let mut hashes = Vec::new();
        for (i, &(t1, f1)) in peaks.iter().enumerate() {
            let partners: Vec<_> = peaks[i + 1..]
                .iter()
                .filter(|&&(t2, _)| t2 > t1 && t2 <= t1 + self.time_window_frames)
                .take(self.fan_out)
                .collect();
            for &(t2, f2) in &partners {
                hashes.push((
                    FingerprintHash { freq1: f1, freq2: *f2, delta_frames: t2 - t1 },
                    t1,
                ));
            }
        }
        hashes
    }

    /// Fingerprint and store a track by its spectrogram.
    pub fn add_track(&mut self, track_id: &str, spectrogram: &Array2<f32>) {
        let peaks = self.extract_peaks(spectrogram);
        let hashes = self.generate_hashes(&peaks);
        for (hash, t) in hashes {
            self.db.entry(hash).or_default().push((track_id.to_string(), t));
        }
    }

    /// Query the database and return the best match.
    pub fn query(&self, spectrogram: &Array2<f32>) -> Option<FingerprintMatch> {
        let peaks = self.extract_peaks(spectrogram);
        let hashes = self.generate_hashes(&peaks);

        // Accumulate score by track_id + time offset
        let mut score_map: HashMap<(String, i32), u32> = HashMap::new();
        for (hash, t_q) in &hashes {
            if let Some(entries) = self.db.get(hash) {
                for (track_id, t_ref) in entries {
                    let offset = *t_ref as i32 - *t_q as i32;
                    *score_map.entry((track_id.clone(), offset)).or_insert(0) += 1;
                }
            }
        }
        if score_map.is_empty() {
            return None;
        }
        let total_hashes = hashes.len().max(1) as f32;
        let ((track_id, offset), count) = score_map
            .into_iter()
            .max_by_key(|(_, c)| *c)?;
        Some(FingerprintMatch {
            track_id,
            score: count as f32 / total_hashes,
            time_offset_frames: offset,
        })
    }

    pub fn db_size(&self) -> usize {
        self.db.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spectrogram(n_frames: usize, n_bins: usize, peak_t: usize, peak_f: usize) -> Array2<f32> {
        let mut spec = Array2::<f32>::zeros((n_frames, n_bins));
        // Background noise
        for t in 0..n_frames {
            for f in 0..n_bins {
                spec[[t, f]] = -50.0;
            }
        }
        // Peaks
        for dt in 0..3 {
            let t = (peak_t + dt * 10).min(n_frames - 2);
            let f = (peak_f + dt * 5).min(n_bins - 2);
            spec[[t, f]] = 0.0;
        }
        spec
    }

    #[test]
    fn test_fingerprint_roundtrip() {
        let mut fp = AcousticFingerprinter::new(-30.0, 5, 20);
        let spec = make_spectrogram(100, 64, 10, 20);
        fp.add_track("track_001", &spec);
        assert!(fp.db_size() > 0, "DB should contain hashes");

        // Query with same spectrogram should return track_001
        let result = fp.query(&spec);
        assert!(result.is_some(), "Should find a match");
        assert_eq!(result.unwrap().track_id, "track_001");
    }
}

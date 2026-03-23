//! MVDR (Minimum Variance Distortionless Response) beamformer.
use nalgebra::{DMatrix, DVector};
use std::f32::consts::PI;

/// MVDR Beamformer result.
#[derive(Debug, Clone)]
pub struct BeamPattern {
    /// Angles in degrees.
    pub angles_deg: Vec<f32>,
    /// Beam power at each angle in dB.
    pub power_db: Vec<f32>,
    /// Estimated bearing of strongest source (degrees).
    pub doa_deg: f32,
}

/// MVDR beamformer for a uniform linear array (ULA).
pub struct MvdrBeamformer {
    pub n_elements: usize,
    pub element_spacing_m: f32,
    pub sound_speed_mps: f32,
    pub diagonal_loading: f32,
    pub angle_resolution_deg: f32,
}

impl MvdrBeamformer {
    pub fn new(
        n_elements: usize,
        element_spacing_m: f32,
        sound_speed_mps: f32,
        diagonal_loading: f32,
        angle_resolution_deg: f32,
    ) -> Self {
        Self {
            n_elements,
            element_spacing_m,
            sound_speed_mps,
            diagonal_loading,
            angle_resolution_deg,
        }
    }

    /// Compute beam pattern from a covariance-like matrix (n_elements × n_elements).
    /// `covariance`: flattened row-major Re/Im pairs or just real diagonal.
    /// For simplicity here we work with real-valued input and synthetic steering.
    /// `signal_matrix`: (n_snapshots, n_elements) matrix of array snapshots.
    pub fn compute(&self, signal_matrix: &[Vec<f32>]) -> BeamPattern {
        let n_snap = signal_matrix.len();
        let n_elem = self.n_elements;
        if n_snap < 2 || signal_matrix[0].len() < n_elem {
            return BeamPattern {
                angles_deg: vec![],
                power_db: vec![],
                doa_deg: 0.0,
            };
        }

        // Build real sample covariance matrix R = (1/N) * X^H * X
        // Using f64 for numerical stability
        let mut r = DMatrix::<f64>::zeros(n_elem, n_elem);
        for snap in signal_matrix {
            for i in 0..n_elem {
                for j in 0..n_elem {
                    r[(i, j)] += snap[i] as f64 * snap[j] as f64;
                }
            }
        }
        r /= n_snap as f64;
        // Diagonal loading for stability
        for i in 0..n_elem {
            r[(i, i)] += self.diagonal_loading as f64;
        }

        // Invert R
        let r_inv = match r.try_inverse() {
            Some(inv) => inv,
            None => DMatrix::identity(n_elem, n_elem),
        };

        // Scan angles
        let n_angles = (180.0 / self.angle_resolution_deg) as usize + 1;
        let mut angles = Vec::with_capacity(n_angles);
        let mut powers = Vec::with_capacity(n_angles);

        for a_idx in 0..n_angles {
            let theta_deg = -90.0 + a_idx as f32 * self.angle_resolution_deg;
            let theta_rad = theta_deg * PI / 180.0;
            // Real-valued steering vector for ULA
            let sv: DVector<f64> = DVector::from_fn(n_elem, |i, _| {
                let tau =
                    i as f32 * self.element_spacing_m * theta_rad.sin() / self.sound_speed_mps;
                (2.0 * PI * 1000.0 * tau).cos() as f64 // 1kHz reference
            });
            // MVDR weight: w = R^{-1} a / (a^H R^{-1} a)
            let r_inv_sv = &r_inv * &sv;
            let denom = sv.dot(&r_inv_sv);
            let power = if denom.abs() > 1e-30 {
                1.0 / denom
            } else {
                0.0
            };
            let power_db = if power > 1e-30 {
                10.0 * power.abs().log10() as f32
            } else {
                -100.0f32
            };
            angles.push(theta_deg);
            powers.push(power_db);
        }

        let doa_idx = powers
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let doa_deg = angles[doa_idx];

        BeamPattern {
            angles_deg: angles,
            power_db: powers,
            doa_deg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beamformer_returns_pattern() {
        let bf = MvdrBeamformer::new(8, 0.037, 1500.0, 0.001, 1.0);
        // Synthesize plane wave arriving from 30 degrees
        let n_snap = 64usize;
        let theta = 30.0f32 * PI / 180.0;
        let snapshots: Vec<Vec<f32>> = (0..n_snap)
            .map(|_| {
                (0..8)
                    .map(|i| {
                        let tau = i as f32 * 0.037 * theta.sin() / 1500.0;
                        (2.0 * PI * 1000.0 * tau).cos()
                    })
                    .collect()
            })
            .collect();
        let pattern = bf.compute(&snapshots);
        assert_eq!(pattern.angles_deg.len(), pattern.power_db.len());
        assert!(!pattern.angles_deg.is_empty());
    }
}

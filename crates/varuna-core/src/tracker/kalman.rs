//! Kalman filter-based target tracker for bearing/range/depth estimates.
use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};

/// State vector: [bearing_deg, bearing_rate, range_m, range_rate].
pub type State = Vector4<f32>;
/// Measurement vector: [bearing_deg, range_m].
pub type Measurement = Vector2<f32>;

/// A single tracked target.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: u32,
    pub state: State,
    pub covariance: Matrix4<f32>,
    pub coasting_frames: u32,
    pub label: Option<String>,
    pub confidence: f32,
}

/// Kalman tracker configuration.
#[derive(Debug, Clone)]
pub struct KalmanConfig {
    pub dt: f32,
    pub process_noise_q: f32,
    pub meas_noise_bearing: f32,
    pub meas_noise_range: f32,
    pub gating_threshold: f32,
    pub max_coasting: u32,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            dt: 0.05,
            process_noise_q: 0.01,
            meas_noise_bearing: 0.5,
            meas_noise_range: 5.0,
            gating_threshold: 9.21,
            max_coasting: 10,
        }
    }
}

/// Multi-target Kalman tracker.
pub struct KalmanTracker {
    pub config: KalmanConfig,
    tracks: Vec<Track>,
    next_id: u32,
    f_mat: Matrix4<f32>,
    h_mat: Matrix2x4<f32>,
    q_mat: Matrix4<f32>,
    r_mat: Matrix2<f32>,
}

impl KalmanTracker {
    pub fn new(config: KalmanConfig) -> Self {
        let dt = config.dt;
        let q = config.process_noise_q;
        let r_b = config.meas_noise_bearing;
        let r_r = config.meas_noise_range;

        // State transition: constant velocity model
        let f_mat = Matrix4::new(
            1.0, dt, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 1.0,
        );

        // Measurement matrix: observe bearing and range
        let h_mat = Matrix2x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        // Process noise covariance
        let q_mat = Matrix4::new(
            q * dt * dt,
            0.0,
            0.0,
            0.0,
            0.0,
            q,
            0.0,
            0.0,
            0.0,
            0.0,
            q * dt * dt,
            0.0,
            0.0,
            0.0,
            0.0,
            q,
        );

        // Measurement noise covariance
        let r_mat = Matrix2::new(r_b, 0.0, 0.0, r_r);

        Self {
            config,
            tracks: Vec::new(),
            next_id: 0,
            f_mat,
            h_mat,
            q_mat,
            r_mat,
        }
    }

    /// Update with new measurements. Returns current active tracks.
    pub fn update(&mut self, measurements: &[(f32, f32)]) -> &[Track] {
        // Predict all tracks
        for track in &mut self.tracks {
            track.state = self.f_mat * track.state;
            track.covariance =
                self.f_mat * track.covariance * self.f_mat.transpose() + self.q_mat;
        }

        let mut assigned = vec![false; measurements.len()];
        // Associate measurements to tracks (nearest neighbour gating)
        for track in &mut self.tracks {
            let mut best_meas = None;
            let mut best_dist = f32::MAX;
            for (j, &(bearing, range)) in measurements.iter().enumerate() {
                if assigned[j] {
                    continue;
                }
                let z = Vector2::new(bearing, range);
                let z_pred = self.h_mat * track.state;
                let innov = z - z_pred;
                let s = self.h_mat * track.covariance * self.h_mat.transpose() + self.r_mat;
                let s_inv = match s.try_inverse() {
                    Some(inv) => inv,
                    None => continue,
                };
                let dist = (innov.transpose() * s_inv * innov)[(0, 0)];
                if dist < self.config.gating_threshold && dist < best_dist {
                    best_dist = dist;
                    best_meas = Some(j);
                }
            }
            if let Some(j) = best_meas {
                assigned[j] = true;
                track.coasting_frames = 0;
                // Kalman update
                let z = Vector2::new(measurements[j].0, measurements[j].1);
                let z_pred = self.h_mat * track.state;
                let innov = z - z_pred;
                let s = self.h_mat * track.covariance * self.h_mat.transpose() + self.r_mat;
                let s_inv = match s.try_inverse() {
                    Some(inv) => inv,
                    None => continue,
                };
                let k_gain: Matrix4x2<f32> = track.covariance * self.h_mat.transpose() * s_inv;
                track.state += k_gain * innov;
                let i_kh = Matrix4::identity() - k_gain * self.h_mat;
                track.covariance = i_kh * track.covariance;
            } else {
                track.coasting_frames += 1;
            }
        }

        // Initialise new tracks for unassigned measurements
        for (j, &(bearing, range)) in measurements.iter().enumerate() {
            if !assigned[j] {
                let state = Vector4::new(bearing, 0.0, range, 0.0);
                let cov = Matrix4::identity() * 100.0;
                self.tracks.push(Track {
                    id: self.next_id,
                    state,
                    covariance: cov,
                    coasting_frames: 0,
                    label: None,
                    confidence: 0.0,
                });
                self.next_id += 1;
            }
        }

        // Prune coasted tracks
        self.tracks
            .retain(|t| t.coasting_frames <= self.config.max_coasting);

        &self.tracks
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_initialises_track() {
        let mut tracker = KalmanTracker::new(KalmanConfig::default());
        let meas = vec![(30.0f32, 1500.0f32)];
        let tracks = tracker.update(&meas);
        assert_eq!(tracks.len(), 1);
        assert!((tracks[0].state[0] - 30.0).abs() < 1.0);
    }

    #[test]
    fn test_tracker_prunes_coasted_tracks() {
        let config = KalmanConfig {
            max_coasting: 2,
            ..Default::default()
        };
        let mut tracker = KalmanTracker::new(config);
        // Init with measurement
        tracker.update(&[(45.0, 1000.0)]);
        // Update with no matching measurement several times
        for _ in 0..5 {
            tracker.update(&[]);
        }
        assert_eq!(tracker.tracks().len(), 0, "Track should have been pruned");
    }
}

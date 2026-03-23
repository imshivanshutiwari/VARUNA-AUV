//! Dataset metadata and stub fetchers for ShipsEar and DeepShip.
use serde::{Deserialize, Serialize};

/// Metadata describing an acoustic dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub description: String,
    pub classes: Vec<String>,
    pub n_samples: usize,
    pub sample_rate_hz: u32,
    pub url: String,
}

impl DatasetInfo {
    /// Return the ShipsEar dataset metadata record.
    pub fn shipsear() -> Self {
        Self {
            name: "ShipsEar".to_string(),
            description: "90 underwater noise recordings of 11 vessel categories at sea."
                .to_string(),
            classes: vec![
                "Background".to_string(),
                "Dinghy".to_string(),
                "Fishboat".to_string(),
                "Motorboat".to_string(),
                "Mussel-boat".to_string(),
                "Natural".to_string(),
                "Oceanliner".to_string(),
                "Passenger".to_string(),
                "RORO".to_string(),
                "Sailboat".to_string(),
                "Tugboat".to_string(),
            ],
            n_samples: 90,
            sample_rate_hz: 52734,
            url: "https://underwaternoise.atlanttic.uvigo.es/".to_string(),
        }
    }

    /// Return the DeepShip dataset metadata record.
    pub fn deepship() -> Self {
        Self {
            name: "DeepShip".to_string(),
            description: "47000+ 5-second segments from real vessel recordings, 4 classes."
                .to_string(),
            classes: vec![
                "Cargo".to_string(),
                "Passengership".to_string(),
                "Tanker".to_string(),
                "Tug".to_string(),
            ],
            n_samples: 47000,
            sample_rate_hz: 22050,
            url: "https://github.com/irfankamboh/DeepShip".to_string(),
        }
    }
}

/// Print ShipsEar dataset metadata (stub – real fetch requires manual download).
pub fn fetch_shipsear_stub() -> DatasetInfo {
    let info = DatasetInfo::shipsear();
    tracing::info!(
        "ShipsEar dataset: {} classes, {} samples",
        info.classes.len(),
        info.n_samples
    );
    info
}

/// Print DeepShip dataset metadata (stub – real fetch requires manual download).
pub fn fetch_deepship_stub() -> DatasetInfo {
    let info = DatasetInfo::deepship();
    tracing::info!(
        "DeepShip dataset: {} classes, {} samples",
        info.classes.len(),
        info.n_samples
    );
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shipsear_metadata() {
        let info = DatasetInfo::shipsear();
        assert_eq!(info.name, "ShipsEar");
        assert_eq!(info.classes.len(), 11);
    }

    #[test]
    fn test_deepship_metadata() {
        let info = DatasetInfo::deepship();
        assert_eq!(info.name, "DeepShip");
        assert_eq!(info.classes.len(), 4);
    }
}

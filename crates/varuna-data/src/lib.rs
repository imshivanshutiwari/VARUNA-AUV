//! varuna-data – audio file I/O and dataset fetching utilities.
pub mod dataset;
pub mod wav;

pub use dataset::{fetch_deepship_stub, fetch_shipsear_stub, DatasetInfo};
pub use wav::{AudioBuffer, WavReader};

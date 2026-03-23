//! varuna-data – audio file I/O and dataset fetching utilities.
pub mod wav;
pub mod dataset;

pub use wav::{WavReader, AudioBuffer};
pub use dataset::{DatasetInfo, fetch_shipsear_stub, fetch_deepship_stub};

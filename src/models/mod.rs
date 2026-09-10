pub mod detector;
pub mod recognizer;

#[cfg(feature = "table")]
pub mod table;

pub use detector::{DetBox, TextDetector};
pub use recognizer::TextRecognizer;

#[cfg(feature = "table")]
pub use table::{CellBox, TableStructurePredictor, TableStructureResult};

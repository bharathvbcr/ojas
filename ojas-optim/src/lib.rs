//! Lane B. AdamW (torch single-tensor order) and Muon NS5.
//!
//! Step counters go through [`ojas_core::next_step`]. Non-finite gradients
//! are refused. No update is implemented in this scaffold.

#[doc(hidden)]
pub use ojas_core::AdamWConfig;

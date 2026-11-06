//! Re-exports the Burn 0.21 surface `arch02-burn` calls.
//!
//! `pub use burn_core::*` matches the crates.io meta-crate. The explicit
//! `prelude` module shadows `burn_core::prelude` so `use burn::prelude::*`
//! also brings `nn` into scope, which is what the upstream prelude does.

pub use burn_core::*;

pub mod backend;

pub mod nn {
    pub use burn_nn::*;
}

pub mod optim {
    pub use burn_optim::*;
}

pub mod prelude {
    pub use burn_core::prelude::*;
    pub use crate::nn;
}

#[cfg(feature = "ndarray")]
pub use burn_ndarray as ndarray;
#[cfg(feature = "ndarray")]
pub use burn_ndarray::NdArray;

#[cfg(feature = "autodiff")]
pub use burn_autodiff as autodiff;
#[cfg(feature = "autodiff")]
pub use burn_autodiff::Autodiff;

#[cfg(feature = "metal")]
pub use burn_wgpu as wgpu;
#[cfg(feature = "metal")]
pub use burn_wgpu::Metal;

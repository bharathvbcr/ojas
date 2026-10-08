//! MLP: up-project → squared leaky-ReLU (slope 0.5) → down-project.
//! f(z) = leaky_relu(z, 0.5)^2 ; up: 512→1536, down: 1536→512 (zero-init).

use burn::module::{Module, Param};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub up_w: Param<Tensor<B, 2>>,   // [512, 1536]
    pub down_w: Param<Tensor<B, 2>>, // [1536, 512] (zero-init)
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        let h = self.up_w.dims()[1];
        let z = x.reshape([b * t, c]).matmul(self.up_w.val());
        // leaky_relu(z, 0.5) = max(z,0) + 0.5*min(z,0), then square
        let a = z.clone().clamp_min(0.0) + z.clamp_max(0.0) * 0.5;
        let a = a.powf_scalar(2.0);
        a.reshape([b * t, h]).matmul(self.down_w.val()).reshape([b, t, c])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    type B = burn::backend::NdArray;

    #[test]
    fn squared_leaky_relu() {
        let device = Default::default();
        // identity up (1x1), identity down: check activation alone
        let mlp = Mlp::<B> {
            up_w: Param::from_tensor(Tensor::from_floats([[1.0]], &device)),
            down_w: Param::from_tensor(Tensor::from_floats([[1.0]], &device)),
        };
        let x = Tensor::<B, 3>::from_floats([[[2.0], [-2.0], [0.0]]], &device);
        let y = mlp.forward(x).into_data();
        let v = y.as_slice::<f32>().unwrap();
        assert!((v[0] - 4.0).abs() < 1e-6); // 2^2
        assert!((v[1] - 1.0).abs() < 1e-6); // (-2*0.5)^2 = 1
        assert!(v[2].abs() < 1e-6);
    }
}

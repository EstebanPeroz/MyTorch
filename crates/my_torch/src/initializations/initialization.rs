use std::fmt::Debug;

use rand_distr::{Distribution, Normal};

use crate::matrix::{Matrix, Scalar};
use crate::rng::Rng;

pub trait Initialization: Debug {
    fn weights(&self, inputs: usize, outputs: usize, rng: &mut Rng) -> Matrix;
}

pub(super) fn normal_weights(
    inputs: usize,
    outputs: usize,
    standard_deviation: impl Fn(Scalar, Scalar) -> Scalar,
    rng: &mut Rng,
) -> Matrix {
    assert!(
        inputs > 0 && outputs > 0,
        "a layer cannot have {inputs} inputs and {outputs} outputs"
    );
    let normal = Normal::new(0.0, standard_deviation(inputs as Scalar, outputs as Scalar))
        .expect("the standard deviation of a non-empty layer is finite");
    let values = (0..inputs * outputs)
        .map(|_| normal.sample(rng.generator()))
        .collect();
    Matrix::from_vec(inputs, outputs, values).expect("the shape is not empty")
}

#[cfg(test)]
pub(super) fn assert_spread(weights: &Matrix, expected_deviation: Scalar) {
    let count = weights.data().len() as Scalar;
    let mean = weights.data().iter().sum::<Scalar>() / count;
    let deviation = (weights.data().iter().map(|w| w * w).sum::<Scalar>() / count).sqrt();
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!(
        (deviation - expected_deviation).abs() < 0.05 * expected_deviation,
        "deviation {deviation}, expected {expected_deviation}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(inputs: usize, outputs: usize, seed: u64) -> Matrix {
        normal_weights(inputs, outputs, |_, _| 0.1, &mut Rng::new(seed))
    }

    #[test]
    fn has_the_layer_shape_and_spread() {
        let weights = weights(200, 100, 1);
        assert_eq!((weights.rows(), weights.cols()), (200, 100));
        assert_spread(&weights, 0.1);
    }

    #[test]
    fn the_same_seed_gives_the_same_weights() {
        assert_eq!(weights(3, 2, 4), weights(3, 2, 4));
        assert_ne!(weights(3, 2, 4), weights(3, 2, 5));
    }

    #[test]
    #[should_panic(expected = "a layer cannot have 0 inputs and 2 outputs")]
    fn panics_on_an_empty_layer() {
        weights(0, 2, 1);
    }
}

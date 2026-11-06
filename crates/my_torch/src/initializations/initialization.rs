use std::fmt::Debug;

use rand_distr::{Distribution, Normal};

use super::he_normal::HeNormal;
use super::xavier_normal::XavierNormal;
use crate::matrix::{Matrix, Scalar};
use crate::names::find_by_name;
use crate::rng::Rng;

pub trait Initialization: Debug {
    fn name(&self) -> &'static str;

    fn weights(&self, inputs: usize, outputs: usize, rng: &mut Rng) -> Matrix;
}

impl<I: Initialization + ?Sized> Initialization for Box<I> {
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    fn weights(&self, inputs: usize, outputs: usize, rng: &mut Rng) -> Matrix {
        self.as_ref().weights(inputs, outputs, rng)
    }
}

pub fn initialization_from_name(name: &str) -> Result<Box<dyn Initialization>, String> {
    let all: Vec<Box<dyn Initialization>> = vec![Box::new(XavierNormal), Box::new(HeNormal)];
    find_by_name("initialization", name, all, |initialization| {
        initialization.name()
    })
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
    fn each_initialization_is_found_by_its_name() {
        for name in ["xavier_normal", "he_normal"] {
            let initialization = initialization_from_name(name).unwrap();
            assert_eq!(initialization.name(), name);
            assert_eq!(
                initialization.weights(3, 2, &mut Rng::new(1)),
                initialization.as_ref().weights(3, 2, &mut Rng::new(1))
            );
        }
        assert_eq!(
            initialization_from_name("he_normal")
                .unwrap()
                .weights(50, 20, &mut Rng::new(2)),
            HeNormal.weights(50, 20, &mut Rng::new(2))
        );
    }

    #[test]
    fn rejects_an_unknown_initialization_name() {
        assert_eq!(
            initialization_from_name("uniform").err(),
            Some(
                "unknown initialization \"uniform\", expected one of: xavier_normal, he_normal"
                    .to_string()
            )
        );
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

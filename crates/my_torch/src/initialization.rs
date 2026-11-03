use crate::matrix::{Matrix, Scalar};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initialization {
    Xavier,
    He,
}

impl Initialization {
    pub fn weights(self, inputs: usize, outputs: usize, rng: &mut StdRng) -> Matrix {
        assert!(
            inputs > 0 && outputs > 0,
            "a layer cannot have {inputs} inputs and {outputs} outputs"
        );
        let normal = Normal::new(0.0, self.standard_deviation(inputs, outputs))
            .expect("the standard deviation of a non-empty layer is finite");
        let values = (0..inputs * outputs).map(|_| normal.sample(rng)).collect();
        Matrix::from_vec(inputs, outputs, values).expect("the shape is not empty")
    }

    fn standard_deviation(self, inputs: usize, outputs: usize) -> Scalar {
        let variance = match self {
            Initialization::Xavier => 2.0 / (inputs + outputs) as Scalar,
            Initialization::He => 2.0 / inputs as Scalar,
        };
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn spread(weights: &Matrix) -> (Scalar, Scalar) {
        let count = weights.data().len() as Scalar;
        let mean = weights.data().iter().sum::<Scalar>() / count;
        let deviation = (weights.data().iter().map(|w| w * w).sum::<Scalar>() / count).sqrt();
        (mean, deviation)
    }

    #[test]
    fn standard_deviations_follow_their_formulas() {
        assert_eq!(Initialization::Xavier.standard_deviation(6, 2), 0.5);
        assert_eq!(Initialization::He.standard_deviation(8, 3), 0.5);
    }

    #[test]
    fn weights_have_the_layer_shape() {
        let weights = Initialization::Xavier.weights(3, 2, &mut StdRng::seed_from_u64(1));
        assert_eq!((weights.rows(), weights.cols()), (3, 2));
    }

    #[test]
    #[should_panic(expected = "a layer cannot have 0 inputs and 2 outputs")]
    fn weights_panic_on_an_empty_layer() {
        Initialization::He.weights(0, 2, &mut StdRng::seed_from_u64(1));
    }

    #[test]
    fn xavier_weights_are_centered_with_the_expected_spread() {
        let weights = Initialization::Xavier.weights(100, 100, &mut StdRng::seed_from_u64(2));
        let (mean, deviation) = spread(&weights);
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((deviation - 0.1).abs() < 0.005, "deviation {deviation}");
    }

    #[test]
    fn he_weights_are_centered_with_the_expected_spread() {
        let weights = Initialization::He.weights(200, 100, &mut StdRng::seed_from_u64(3));
        let (mean, deviation) = spread(&weights);
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((deviation - 0.1).abs() < 0.005, "deviation {deviation}");
    }

    #[test]
    fn the_same_seed_gives_the_same_weights() {
        let first = Initialization::He.weights(3, 2, &mut StdRng::seed_from_u64(4));
        let second = Initialization::He.weights(3, 2, &mut StdRng::seed_from_u64(4));
        let other = Initialization::He.weights(3, 2, &mut StdRng::seed_from_u64(5));
        assert_eq!(first, second);
        assert_ne!(first, other);
    }
}

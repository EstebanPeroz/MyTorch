use crate::activations::Activation;
use crate::initializations::Initialization;
use crate::matrix::{Matrix, Scalar};
use crate::rng::Rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    weights: Matrix,
    biases: Matrix,
    activation: Activation,
}

#[derive(Debug, Clone)]
pub struct DenseCache {
    input: Matrix,
    z: Matrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseGradients {
    weights: Matrix,
    biases: Matrix,
}

// Same as Torch "linear" layer, but Keras layer name is Dense, and it includes the activation function (same as we did here)
impl Dense {
    pub fn new(weights: Matrix, biases: Matrix, activation: Activation) -> Result<Dense, String> {
        if biases.rows() != 1 || biases.cols() != weights.cols() {
            return Err(format!(
                "a layer with {} outputs needs 1 × {} biases, got a {biases}",
                weights.cols(),
                weights.cols()
            ));
        }
        Ok(Dense {
            weights,
            biases,
            activation,
        })
    }

    pub fn random(
        inputs: usize,
        outputs: usize,
        activation: Activation,
        initialization: &impl Initialization,
        rng: &mut Rng,
    ) -> Dense {
        Dense {
            weights: initialization.weights(inputs, outputs, rng),
            biases: Matrix::zeros(1, outputs),
            activation,
        }
    }

    pub fn input_size(&self) -> usize {
        self.weights.rows()
    }

    pub fn output_size(&self) -> usize {
        self.weights.cols()
    }

    pub fn weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn biases(&self) -> &Matrix {
        &self.biases
    }

    pub fn activation(&self) -> Activation {
        self.activation
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.activation.apply(&self.weighted_sum(input))
    }

    pub fn forward_cached(&self, input: &Matrix) -> (Matrix, DenseCache) {
        let z = self.weighted_sum(input);
        let output = self.activation.apply(&z);
        let cache = DenseCache {
            input: input.clone(),
            z,
        };
        (output, cache)
    }

    pub fn backward(
        &self,
        cache: &DenseCache,
        output_gradient: &Matrix,
    ) -> (Matrix, DenseGradients) {
        assert!(
            output_gradient.rows() == cache.z.rows()
                && output_gradient.cols() == self.output_size(),
            "a layer with {} outputs on {} samples cannot take a {output_gradient} as its gradient",
            self.output_size(),
            cache.z.rows()
        );
        let z_gradient =
            output_gradient.zip_map(&self.activation.derivative(&cache.z), |g, d| g * d);
        let gradients = DenseGradients {
            weights: cache.input.transpose_matmul(&z_gradient),
            biases: z_gradient.sum_rows(),
        };
        (z_gradient.matmul_transpose(&self.weights), gradients)
    }

    pub fn update(&mut self, gradients: &DenseGradients, rule: impl Fn(Scalar, Scalar) -> Scalar) {
        self.weights = self.weights.zip_map(&gradients.weights, &rule);
        self.biases = self.biases.zip_map(&gradients.biases, &rule);
    }

    fn weighted_sum(&self, input: &Matrix) -> Matrix {
        assert_eq!(
            input.cols(),
            self.input_size(),
            "a layer with {} inputs cannot read a {input}",
            self.input_size()
        );
        input.matmul(&self.weights).add_row(&self.biases)
    }
}

impl DenseGradients {
    pub fn weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn biases(&self) -> &Matrix {
        &self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initializations::XavierNormal;

    fn matrix(rows: &[&[Scalar]]) -> Matrix {
        Matrix::from_rows(rows).unwrap()
    }

    fn layer(activation: Activation) -> Dense {
        Dense::new(
            matrix(&[&[1.0, -1.0], &[2.0, 0.5]]),
            matrix(&[&[0.5, -3.0]]),
            activation,
        )
        .unwrap()
    }

    fn nudged(values: &Matrix, row: usize, col: usize, step: Scalar) -> Matrix {
        let mut values = values.clone();
        values.set(row, col, values.get(row, col) + step);
        values
    }

    fn assert_close(numeric: Scalar, analytic: Scalar, what: &str) {
        let tolerance = 1e-2 * (1.0 + analytic.abs());
        assert!(
            (numeric - analytic).abs() < tolerance,
            "{what}: measured {numeric}, backward gave {analytic}"
        );
    }

    #[test]
    fn new_rejects_biases_of_the_wrong_shape() {
        assert_eq!(
            Dense::new(Matrix::zeros(2, 3), Matrix::zeros(2, 3), Activation::Relu),
            Err("a layer with 3 outputs needs 1 × 3 biases, got a 2 × 3 matrix".to_string())
        );
        assert_eq!(
            Dense::new(Matrix::zeros(2, 3), Matrix::zeros(1, 2), Activation::Relu),
            Err("a layer with 3 outputs needs 1 × 3 biases, got a 1 × 2 matrix".to_string())
        );
    }

    #[test]
    fn random_draws_weights_and_zero_biases() {
        let dense = Dense::random(3, 2, Activation::Tanh, &XavierNormal, &mut Rng::new(5));
        assert_eq!((dense.input_size(), dense.output_size()), (3, 2));
        assert_eq!(dense.biases(), &Matrix::zeros(1, 2));
        assert_eq!(dense.activation(), Activation::Tanh);
        assert!(dense.weights().data().iter().any(|&w| w != 0.0));
    }

    #[test]
    fn forward_computes_one_output_row_per_input_row() {
        let input = matrix(&[&[1.0, 1.0], &[0.0, 2.0]]);
        assert_eq!(
            layer(Activation::Identity).forward(&input),
            matrix(&[&[3.5, -3.5], &[4.5, -2.0]])
        );
        assert_eq!(
            layer(Activation::Relu).forward(&input),
            matrix(&[&[3.5, 0.0], &[4.5, 0.0]])
        );
    }

    #[test]
    #[should_panic(expected = "a layer with 2 inputs cannot read a 1 × 3 matrix")]
    fn forward_panics_on_the_wrong_number_of_inputs() {
        layer(Activation::Identity).forward(&Matrix::zeros(1, 3));
    }

    #[test]
    fn forward_cached_gives_the_same_output() {
        let input = matrix(&[&[1.0, 1.0], &[0.0, 2.0]]);
        let dense = layer(Activation::Sigmoid);
        assert_eq!(dense.forward_cached(&input).0, dense.forward(&input));
    }

    #[test]
    fn backward_follows_the_chain_rule() {
        let dense = layer(Activation::Relu);
        let input = matrix(&[&[1.0, 1.0], &[0.0, 2.0]]);
        let (_, cache) = dense.forward_cached(&input);
        let (input_gradient, gradients) =
            dense.backward(&cache, &matrix(&[&[1.0, 5.0], &[2.0, 5.0]]));
        assert_eq!(gradients.weights(), &matrix(&[&[1.0, 0.0], &[5.0, 0.0]]));
        assert_eq!(gradients.biases(), &matrix(&[&[3.0, 0.0]]));
        assert_eq!(input_gradient, matrix(&[&[1.0, 2.0], &[2.0, 4.0]]));
    }

    #[test]
    #[should_panic(
        expected = "a layer with 2 outputs on 2 samples cannot take a 2 × 3 matrix as its gradient"
    )]
    fn backward_panics_on_a_gradient_of_the_wrong_shape() {
        let dense = layer(Activation::Identity);
        let (_, cache) = dense.forward_cached(&Matrix::zeros(2, 2));
        dense.backward(&cache, &Matrix::zeros(2, 3));
    }

    #[test]
    fn backward_matches_finite_differences() {
        let step = 1e-2;
        let dense = Dense::new(
            matrix(&[&[0.3, -0.8, 0.5], &[0.9, 0.1, -0.4]]),
            matrix(&[&[0.1, -0.2, 0.3]]),
            Activation::Tanh,
        )
        .unwrap();
        let input = matrix(&[&[0.5, -1.0], &[1.5, 0.2], &[-0.3, 0.7]]);
        let output_gradient = matrix(&[&[1.0, -2.0, 0.5], &[0.3, 1.0, -1.0], &[2.0, 0.0, 1.0]]);
        let loss = |layer: &Dense, input: &Matrix| -> Scalar {
            layer
                .forward(input)
                .zip_map(&output_gradient, |a, g| a * g)
                .data()
                .iter()
                .sum()
        };
        let slope = |above: Scalar, below: Scalar| (above - below) / (2.0 * step);
        let (_, cache) = dense.forward_cached(&input);
        let (input_gradient, gradients) = dense.backward(&cache, &output_gradient);

        for row in 0..2 {
            for col in 0..3 {
                let shifted = |step| {
                    let weights = nudged(dense.weights(), row, col, step);
                    let layer = Dense::new(weights, dense.biases().clone(), Activation::Tanh);
                    loss(&layer.unwrap(), &input)
                };
                let numeric = slope(shifted(step), shifted(-step));
                assert_close(numeric, gradients.weights().get(row, col), "weight");
            }
        }
        for col in 0..3 {
            let shifted = |step| {
                let biases = nudged(dense.biases(), 0, col, step);
                let layer = Dense::new(dense.weights().clone(), biases, Activation::Tanh);
                loss(&layer.unwrap(), &input)
            };
            let numeric = slope(shifted(step), shifted(-step));
            assert_close(numeric, gradients.biases().get(0, col), "bias");
        }
        for row in 0..3 {
            for col in 0..2 {
                let shifted = |step| loss(&dense, &nudged(&input, row, col, step));
                let numeric = slope(shifted(step), shifted(-step));
                assert_close(numeric, input_gradient.get(row, col), "input");
            }
        }
    }

    #[test]
    fn update_applies_the_rule_to_every_parameter() {
        let mut dense = layer(Activation::Identity);
        let gradients = DenseGradients {
            weights: matrix(&[&[1.0, 1.0], &[1.0, 1.0]]),
            biases: matrix(&[&[2.0, 2.0]]),
        };
        dense.update(&gradients, |parameter, gradient| parameter - 0.5 * gradient);
        assert_eq!(dense.weights(), &matrix(&[&[0.5, -1.5], &[1.5, 0.0]]));
        assert_eq!(dense.biases(), &matrix(&[&[-0.5, -4.0]]));
    }
}

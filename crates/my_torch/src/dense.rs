use crate::activations::Activation;
use crate::initializations::Initialization;
use crate::matrix::{Matrix, Scalar};
use crate::rng::Rng;
use rand::RngExt;

#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    weights: Matrix,
    biases: Matrix,
    activation: Activation,
    /// Share of the outputs switched off at each training step. It only matters while training.
    dropout: Scalar,
}

#[derive(Debug, Clone)]
pub struct DenseCache {
    input: Matrix,
    z: Matrix,
    /// The dropout mask of the step: 0 for a switched off output, `1 / (1 - dropout)` for the others.
    mask: Option<Matrix>,
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
            dropout: 0.0,
        })
    }

    /// The same layer, with a share `rate` of its outputs randomly switched off at each training step
    /// (inverted dropout: the others are scaled up, so predictions need no correction).
    pub fn with_dropout(mut self, rate: Scalar) -> Result<Dense, String> {
        if !(0.0..1.0).contains(&rate) {
            return Err(format!(
                "the dropout rate must be at least 0 and less than 1, got {rate}"
            ));
        }
        self.dropout = rate;
        Ok(self)
    }

    pub fn random(
        inputs: usize,
        outputs: usize,
        activation: Activation,
        initialization: &dyn Initialization,
        rng: &mut Rng,
    ) -> Dense {
        Dense {
            weights: initialization.weights(inputs, outputs, rng),
            biases: Matrix::zeros(1, outputs),
            activation,
            dropout: 0.0,
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

    pub fn dropout(&self) -> Scalar {
        self.dropout
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
            mask: None,
        };
        (output, cache)
    }

    /// Like `forward_cached`, with dropout: each output of each sample is switched off with probability
    /// `dropout`, drawn from `rng`. A layer without dropout draws nothing.
    pub fn forward_training(&self, input: &Matrix, rng: &mut Rng) -> (Matrix, DenseCache) {
        let (output, mut cache) = self.forward_cached(input);
        if self.dropout == 0.0 {
            return (output, cache);
        }
        let keep = 1.0 - self.dropout;
        let values = (0..output.rows() * output.cols())
            .map(|_| {
                if rng.generator().random::<Scalar>() < keep {
                    1.0 / keep
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Matrix::from_vec(output.rows(), output.cols(), values)
            .expect("one mask value per output");
        let output = output.zip_map(&mask, |value, factor| value * factor);
        cache.mask = Some(mask);
        (output, cache)
    }

    pub fn backward(
        &self,
        cache: &DenseCache,
        output_gradient: &Matrix,
    ) -> (Matrix, DenseGradients) {
        let (z_gradient, gradients) = self.parameter_gradients(cache, output_gradient);
        (z_gradient.matmul_transpose(&self.weights), gradients)
    }

    /// Like `backward`, without the gradient of the input: for the first layer, where nothing reads it.
    pub fn backward_parameters(
        &self,
        cache: &DenseCache,
        output_gradient: &Matrix,
    ) -> DenseGradients {
        self.parameter_gradients(cache, output_gradient).1
    }

    fn parameter_gradients(
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
        let output_gradient = match &cache.mask {
            Some(mask) => &output_gradient.zip_map(mask, |g, factor| g * factor),
            None => output_gradient,
        };
        let z_gradient =
            output_gradient.zip_map(&self.activation.derivative(&cache.z), |g, d| g * d);
        let gradients = DenseGradients {
            weights: cache.input.transpose_matmul(&z_gradient),
            biases: z_gradient.sum_rows(),
        };
        (z_gradient, gradients)
    }

    pub fn update(&mut self, gradients: &DenseGradients, rule: impl Fn(Scalar, Scalar) -> Scalar) {
        self.weights.zip_apply(&gradients.weights, &rule);
        self.biases.zip_apply(&gradients.biases, &rule);
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
    pub(crate) fn scale(&mut self, factor: Scalar) {
        self.weights.apply(|g| g * factor);
        self.biases.apply(|g| g * factor);
    }

    pub(crate) fn add_scaled(&mut self, other: &DenseGradients, factor: Scalar) {
        self.weights
            .zip_apply(&other.weights, |a, b| a + factor * b);
        self.biases.zip_apply(&other.biases, |a, b| a + factor * b);
    }

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

    #[test]
    fn dropout_is_checked_and_starts_at_zero() {
        assert_eq!(layer(Activation::Relu).dropout(), 0.0);
        assert_eq!(
            layer(Activation::Relu).with_dropout(0.3).unwrap().dropout(),
            0.3
        );
        for rate in [-0.1, 1.0, 2.0, Scalar::NAN] {
            assert!(
                layer(Activation::Relu).with_dropout(rate).is_err(),
                "{rate}"
            );
        }
    }

    #[test]
    fn training_without_dropout_draws_nothing_and_matches_forward() {
        let input = matrix(&[&[1.0, 1.0], &[0.0, 2.0]]);
        let dense = layer(Activation::Relu);
        let mut rng = Rng::new(1);
        assert_eq!(
            dense.forward_training(&input, &mut rng).0,
            dense.forward(&input)
        );
        let mut untouched = Rng::new(1);
        assert_eq!(
            rng.generator().random::<u64>(),
            untouched.generator().random::<u64>()
        );
    }

    #[test]
    fn dropout_switches_off_a_share_of_the_outputs_and_scales_the_others() {
        let dense = Dense::random(
            4,
            200,
            Activation::Identity,
            &XavierNormal,
            &mut Rng::new(2),
        )
        .with_dropout(0.25)
        .unwrap();
        let input = Matrix::from_vec(50, 4, vec![1.0; 200]).unwrap();
        let plain = dense.forward(&input);
        let (dropped, _) = dense.forward_training(&input, &mut Rng::new(3));
        let mut off = 0;
        for (a, b) in plain.data().iter().zip(dropped.data()) {
            if *b == 0.0 {
                off += 1;
            } else {
                assert!((b - a / 0.75).abs() < 1e-4, "{a} became {b}");
            }
        }
        let share = off as Scalar / plain.data().len() as Scalar;
        assert!(
            (share - 0.25).abs() < 0.03,
            "{share} of the outputs were switched off"
        );
    }

    #[test]
    fn the_gradient_does_not_flow_through_switched_off_outputs() {
        let dense = Dense::random(3, 40, Activation::Tanh, &XavierNormal, &mut Rng::new(4))
            .with_dropout(0.5)
            .unwrap();
        let input = Matrix::from_vec(2, 3, vec![0.5, -1.0, 2.0, 1.0, 0.3, -0.7]).unwrap();
        let (output, cache) = dense.forward_training(&input, &mut Rng::new(5));
        let ones = Matrix::from_vec(2, 40, vec![1.0; 80]).unwrap();
        let (_, gradients) = dense.backward(&cache, &ones);
        let mut off = 0;
        for column in 0..40 {
            if output.get(0, column) == 0.0 && output.get(1, column) == 0.0 {
                off += 1;
                assert!(gradients.biases().get(0, column) == 0.0, "column {column}");
                for row in 0..3 {
                    assert!(gradients.weights().get(row, column) == 0.0);
                }
            }
        }
        assert!(off > 0, "no neuron was switched off for both samples");
    }
}

use crate::dense::{Dense, DenseGradients};
use crate::losses::Loss;
use crate::matrix::{Matrix, Scalar};

#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    layers: Vec<Dense>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gradients {
    layers: Vec<DenseGradients>,
}

impl Network {
    pub fn new(layers: Vec<Dense>) -> Result<Network, String> {
        if layers.is_empty() {
            return Err("a network needs at least one layer".into());
        }
        for (index, pair) in layers.windows(2).enumerate() {
            if pair[0].output_size() != pair[1].input_size() {
                return Err(format!(
                    "layer {} has {} outputs but layer {} expects {} inputs",
                    index + 1,
                    pair[0].output_size(),
                    index + 2,
                    pair[1].input_size()
                ));
            }
        }
        Ok(Network { layers })
    }

    pub fn input_size(&self) -> usize {
        self.layers[0].input_size()
    }

    pub fn output_size(&self) -> usize {
        self.output_layer().output_size()
    }

    pub fn output_layer(&self) -> &Dense {
        &self.layers[self.layers.len() - 1]
    }

    pub fn layers(&self) -> &[Dense] {
        &self.layers
    }

    pub fn predict(&self, inputs: &Matrix) -> Result<Matrix, String> {
        self.check_inputs(inputs)?;
        Ok(self
            .layers
            .iter()
            .fold(inputs.clone(), |values, layer| layer.forward(&values)))
    }

    pub fn gradients(
        &self,
        inputs: &Matrix,
        targets: &Matrix,
        loss: &dyn Loss,
    ) -> Result<(Scalar, Gradients), String> {
        self.check_inputs(inputs)?;
        self.check_targets(inputs, targets)?;
        let mut values = inputs.clone();
        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (output, cache) = layer.forward_cached(&values);
            caches.push(cache);
            values = output;
        }
        let value = loss.value(&values, targets);
        let mut gradient = loss.gradient(&values, targets);
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer, cache) in self.layers.iter().zip(&caches).rev() {
            let (input_gradient, layer_gradients) = layer.backward(cache, &gradient);
            layers.push(layer_gradients);
            gradient = input_gradient;
        }
        layers.reverse();
        Ok((value, Gradients { layers }))
    }

    pub fn update(&mut self, gradients: &Gradients, rule: impl Fn(Scalar, Scalar) -> Scalar) {
        assert_eq!(
            gradients.layers.len(),
            self.layers.len(),
            "a network with {} layers cannot take gradients for {}",
            self.layers.len(),
            gradients.layers.len()
        );
        for (layer, layer_gradients) in self.layers.iter_mut().zip(&gradients.layers) {
            layer.update(layer_gradients, &rule);
        }
    }

    fn check_inputs(&self, inputs: &Matrix) -> Result<(), String> {
        if inputs.cols() != self.input_size() {
            return Err(format!(
                "the network expects {} inputs, got {}",
                self.input_size(),
                inputs.cols()
            ));
        }
        Ok(())
    }

    fn check_targets(&self, inputs: &Matrix, targets: &Matrix) -> Result<(), String> {
        if targets.cols() != self.output_size() {
            return Err(format!(
                "the network has {} outputs, but the targets have {} values",
                self.output_size(),
                targets.cols()
            ));
        }
        if targets.rows() != inputs.rows() {
            return Err(format!(
                "{} samples but {} targets",
                inputs.rows(),
                targets.rows()
            ));
        }
        Ok(())
    }
}

impl Gradients {
    pub fn layers(&self) -> &[DenseGradients] {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::initializations::XavierNormal;
    use crate::losses::{MeanSquaredError, SoftmaxCrossEntropy};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};

    fn network(inputs: usize, layers: &[(usize, Activation)], seed: u64) -> Network {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut size = inputs;
        let layers = layers
            .iter()
            .map(|&(outputs, activation)| {
                let layer = Dense::random(size, outputs, activation, &XavierNormal, &mut rng);
                size = outputs;
                layer
            })
            .collect();
        Network::new(layers).unwrap()
    }

    fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Matrix {
        let values = (0..rows * cols)
            .map(|_| StandardNormal.sample(&mut *rng))
            .collect();
        Matrix::from_vec(rows, cols, values).unwrap()
    }

    fn one_hot(rows: usize, cols: usize, rng: &mut StdRng) -> Matrix {
        let mut targets = Matrix::zeros(rows, cols);
        for row in 0..rows {
            targets.set(row, rng.random_range(0..cols), 1.0);
        }
        targets
    }

    fn nudged(
        network: &Network,
        layer: usize,
        bias: bool,
        row: usize,
        col: usize,
        step: Scalar,
    ) -> Network {
        let mut layers = network.layers().to_vec();
        let original = &layers[layer];
        let mut weights = original.weights().clone();
        let mut biases = original.biases().clone();
        let parameter = if bias { &mut biases } else { &mut weights };
        parameter.set(row, col, parameter.get(row, col) + step);
        layers[layer] = Dense::new(weights, biases, original.activation()).unwrap();
        Network::new(layers).unwrap()
    }

    fn assert_gradients_match_finite_differences(
        network: &Network,
        inputs: &Matrix,
        targets: &Matrix,
        loss: &dyn Loss,
    ) {
        let step = 1e-2;
        let (_, gradients) = network.gradients(inputs, targets, loss).unwrap();
        let loss_of = |candidate: &Network| candidate.gradients(inputs, targets, loss).unwrap().0;
        for (index, layer_gradients) in gradients.layers().iter().enumerate() {
            for (bias, analytic) in [
                (false, layer_gradients.weights()),
                (true, layer_gradients.biases()),
            ] {
                for row in 0..analytic.rows() {
                    for col in 0..analytic.cols() {
                        let above = loss_of(&nudged(network, index, bias, row, col, step));
                        let below = loss_of(&nudged(network, index, bias, row, col, -step));
                        let numeric = (above - below) / (2.0 * step);
                        let expected = analytic.get(row, col);
                        assert!(
                            (numeric - expected).abs() <= 1e-3 + 1e-2 * expected.abs(),
                            "layer {index}, bias {bias}, ({row}, {col}): backpropagation {expected}, measured {numeric}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn new_checks_that_layer_sizes_chain() {
        let mut rng = StdRng::seed_from_u64(1);
        let first = Dense::random(3, 4, Activation::Relu, &XavierNormal, &mut rng);
        let second = Dense::random(5, 2, Activation::Identity, &XavierNormal, &mut rng);
        assert_eq!(
            Network::new(vec![first, second]),
            Err("layer 1 has 4 outputs but layer 2 expects 5 inputs".to_string())
        );
        assert_eq!(
            Network::new(vec![]),
            Err("a network needs at least one layer".to_string())
        );
    }

    #[test]
    fn sizes_come_from_the_first_and_last_layers() {
        let network = network(3, &[(4, Activation::Relu), (2, Activation::Identity)], 1);
        assert_eq!((network.input_size(), network.output_size()), (3, 2));
        assert_eq!(network.layers().len(), 2);
        assert_eq!(network.output_layer(), &network.layers()[1]);
    }

    #[test]
    fn predict_runs_every_layer_in_order() {
        let network = network(3, &[(4, Activation::Tanh), (2, Activation::Sigmoid)], 2);
        let inputs = random_matrix(4, 3, &mut StdRng::seed_from_u64(3));
        let [first, second] = network.layers() else {
            panic!("two layers");
        };
        assert_eq!(
            network.predict(&inputs).unwrap(),
            second.forward(&first.forward(&inputs))
        );
    }

    #[test]
    fn predict_gives_one_output_row_per_input_row() {
        let network = network(3, &[(4, Activation::Tanh), (2, Activation::Identity)], 2);
        let inputs = random_matrix(4, 3, &mut StdRng::seed_from_u64(3));
        let together = network.predict(&inputs).unwrap();
        assert_eq!((together.rows(), together.cols()), (4, 2));
        for row in 0..4 {
            let alone = network.predict(&inputs.select_rows(&[row])).unwrap();
            assert_eq!(alone.row(0), together.row(row));
        }
    }

    #[test]
    fn rejects_inputs_and_targets_of_the_wrong_size() {
        let network = network(3, &[(2, Activation::Identity)], 4);
        assert_eq!(
            network.predict(&Matrix::zeros(1, 5)),
            Err("the network expects 3 inputs, got 5".to_string())
        );
        let gradients = |inputs: Matrix, targets: Matrix| {
            network
                .gradients(&inputs, &targets, &MeanSquaredError)
                .map(|_| ())
        };
        assert_eq!(
            gradients(Matrix::zeros(1, 5), Matrix::zeros(1, 2)),
            Err("the network expects 3 inputs, got 5".to_string())
        );
        assert_eq!(
            gradients(Matrix::zeros(1, 3), Matrix::zeros(1, 6)),
            Err("the network has 2 outputs, but the targets have 6 values".to_string())
        );
        assert_eq!(
            gradients(Matrix::zeros(3, 3), Matrix::zeros(2, 2)),
            Err("3 samples but 2 targets".to_string())
        );
    }

    #[test]
    fn backpropagation_matches_finite_differences() {
        let mut rng = StdRng::seed_from_u64(11);
        let inputs = random_matrix(5, 3, &mut rng);
        let targets = one_hot(5, 2, &mut rng);
        let classifier = network(
            3,
            &[
                (4, Activation::Tanh),
                (3, Activation::Sigmoid),
                (2, Activation::Identity),
            ],
            12,
        );
        assert_gradients_match_finite_differences(
            &classifier,
            &inputs,
            &targets,
            &SoftmaxCrossEntropy,
        );
        let regressor = network(
            3,
            &[
                (4, Activation::Sigmoid),
                (3, Activation::Tanh),
                (2, Activation::Sigmoid),
            ],
            13,
        );
        assert_gradients_match_finite_differences(&regressor, &inputs, &targets, &MeanSquaredError);
    }

    #[test]
    fn update_moves_every_layer() {
        let mut network = network(3, &[(4, Activation::Tanh), (2, Activation::Identity)], 5);
        let before = network.clone();
        let (_, gradients) = network
            .gradients(
                &Matrix::zeros(1, 3),
                &Matrix::zeros(1, 2),
                &MeanSquaredError,
            )
            .unwrap();
        network.update(&gradients, |parameter, _| parameter + 1.0);
        for (after, before) in network.layers().iter().zip(before.layers()) {
            assert_eq!(after.weights(), &before.weights().map(|w| w + 1.0));
            assert_eq!(after.biases(), &before.biases().map(|b| b + 1.0));
        }
    }

    #[test]
    #[should_panic(expected = "a network with 1 layers cannot take gradients for 2")]
    fn update_panics_on_gradients_of_another_network() {
        let small = network(3, &[(2, Activation::Identity)], 6);
        let big = network(3, &[(4, Activation::Tanh), (2, Activation::Identity)], 7);
        let (_, gradients) = big
            .gradients(
                &Matrix::zeros(1, 3),
                &Matrix::zeros(1, 2),
                &MeanSquaredError,
            )
            .unwrap();
        small.clone().update(&gradients, |parameter, _| parameter);
    }
}

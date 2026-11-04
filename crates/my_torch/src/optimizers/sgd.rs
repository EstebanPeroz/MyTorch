use super::optimizer::Optimizer;
use crate::matrix::Scalar;
use crate::network::{Gradients, Network};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgd {
    learning_rate: Scalar,
}

impl Sgd {
    pub fn new(learning_rate: Scalar) -> Result<Sgd, String> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(format!(
                "the learning rate must be a positive number, got {learning_rate}"
            ));
        }
        Ok(Sgd { learning_rate })
    }

    pub fn learning_rate(&self) -> Scalar {
        self.learning_rate
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, network: &mut Network, gradients: &Gradients) {
        let rate = self.learning_rate;
        network.update(gradients, |parameter, gradient| parameter - rate * gradient);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::dense::Dense;
    use crate::losses::MeanSquaredError;
    use crate::matrix::Matrix;

    #[test]
    fn rejects_learning_rates_that_are_not_positive_numbers() {
        for rate in [0.0, -0.1, Scalar::NAN, Scalar::INFINITY] {
            assert_eq!(
                Sgd::new(rate),
                Err(format!(
                    "the learning rate must be a positive number, got {rate}"
                ))
            );
        }
        assert_eq!(Sgd::new(0.25).unwrap().learning_rate(), 0.25);
    }

    #[test]
    fn step_moves_parameters_against_the_gradient() {
        let layer = Dense::new(
            Matrix::from_rows(&[[1.0]]).unwrap(),
            Matrix::from_rows(&[[0.0]]).unwrap(),
            Activation::Identity,
        )
        .unwrap();
        let mut network = Network::new(vec![layer]).unwrap();
        let inputs = Matrix::from_rows(&[[2.0]]).unwrap();
        let targets = Matrix::from_rows(&[[0.0]]).unwrap();
        let (_, gradients) = network
            .gradients(&inputs, &targets, &MeanSquaredError)
            .unwrap();
        Sgd::new(0.125).unwrap().step(&mut network, &gradients);
        assert_eq!(network.layers()[0].weights().get(0, 0), 0.0);
        assert_eq!(network.layers()[0].biases().get(0, 0), -0.5);
    }
}

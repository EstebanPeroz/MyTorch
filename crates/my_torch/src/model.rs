use crate::dataset::Dataset;
use crate::matrix::Scalar;
use crate::network::Network;
use crate::rng::Rng;
use crate::trainer::Trainer;

pub struct Model {
    network: Network,
    trainer: Trainer,
    seed: u64,
}

impl Model {
    pub fn new(network: Network, trainer: Trainer, seed: u64) -> Result<Model, String> {
        trainer.check_output_layer(&network)?;
        Ok(Model {
            network,
            trainer,
            seed,
        })
    }

    pub fn network(&self) -> &Network {
        &self.network
    }

    pub fn trainer(&self) -> &Trainer {
        &self.trainer
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn train(&mut self, data: &Dataset) -> Result<Vec<Scalar>, String> {
        self.trainer
            .fit(&mut self.network, data, &mut Rng::new(self.seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::architecture::{Architecture, LayerSpec};
    use crate::initializations::XavierNormal;
    use crate::losses::{MeanSquaredError, SoftmaxCrossEntropy};
    use crate::matrix::Matrix;
    use crate::optimizers::Sgd;

    fn network(output: Activation) -> Network {
        Architecture::new(
            2,
            vec![
                LayerSpec::new(3, Activation::Tanh, XavierNormal).unwrap(),
                LayerSpec::new(2, output, XavierNormal).unwrap(),
            ],
        )
        .unwrap()
        .build(&mut Rng::new(1))
    }

    fn trainer() -> Trainer {
        Trainer::new(SoftmaxCrossEntropy, Sgd::new(0.1).unwrap(), 2, 5).unwrap()
    }

    fn data() -> Dataset {
        Dataset::new(
            Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).unwrap(),
            Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn refuses_a_loss_that_does_not_fit_the_output_layer() {
        assert_eq!(
            Model::new(network(Activation::Sigmoid), trainer(), 3).err(),
            Some(
                "this loss needs raw scores, so the output layer must be Identity, not Sigmoid"
                    .to_string()
            )
        );
        let mse = Trainer::new(MeanSquaredError, Sgd::new(0.1).unwrap(), 2, 5).unwrap();
        assert!(Model::new(network(Activation::Sigmoid), mse, 3).is_ok());
    }

    #[test]
    fn keeps_its_network_trainer_and_seed() {
        let model = Model::new(network(Activation::Identity), trainer(), 3).unwrap();
        assert_eq!(model.network(), &network(Activation::Identity));
        assert_eq!(model.trainer().loss().name(), "softmax_cross_entropy");
        assert_eq!(model.trainer().optimizer().name(), "sgd");
        assert_eq!(
            (model.trainer().batch_size(), model.trainer().epochs()),
            (2, 5)
        );
        assert_eq!(model.seed(), 3);
    }

    #[test]
    fn trains_its_network_with_its_own_seed() {
        let mut model = Model::new(network(Activation::Identity), trainer(), 3).unwrap();
        let history = model.train(&data()).unwrap();
        let mut expected = network(Activation::Identity);
        let expected_history = trainer()
            .fit(&mut expected, &data(), &mut Rng::new(3))
            .unwrap();
        assert_eq!(history, expected_history);
        assert_eq!(model.network(), &expected);
        assert_ne!(model.network(), &network(Activation::Identity));
    }
}

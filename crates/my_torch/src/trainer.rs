use rand::seq::SliceRandom;

use crate::activations::Activation;
use crate::dataset::Dataset;
use crate::losses::Loss;
use crate::matrix::Scalar;
use crate::network::Network;
use crate::optimizers::Optimizer;
use crate::rng::Rng;

pub struct Trainer {
    loss: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
    batch_size: usize,
    epochs: usize,
}

impl Trainer {
    pub fn new(
        loss: impl Loss + 'static,
        optimizer: impl Optimizer + 'static,
        batch_size: usize,
        epochs: usize,
    ) -> Result<Trainer, String> {
        if batch_size == 0 {
            return Err("the batch size must be at least 1".into());
        }
        if epochs == 0 {
            return Err("training needs at least one epoch".into());
        }
        Ok(Trainer {
            loss: Box::new(loss),
            optimizer: Box::new(optimizer),
            batch_size,
            epochs,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn fit(
        &mut self,
        network: &mut Network,
        data: &Dataset,
        rng: &mut Rng,
    ) -> Result<Vec<Scalar>, String> {
        self.check_output_layer(network)?;
        let mut order: Vec<usize> = (0..data.sample_count()).collect();
        let mut history = Vec::with_capacity(self.epochs);
        for _ in 0..self.epochs {
            order.shuffle(rng.generator());
            let mut total = 0.0;
            for indices in order.chunks(self.batch_size) {
                let batch = data.select(indices);
                let (value, gradients) =
                    network.gradients(batch.inputs(), batch.targets(), self.loss.as_ref())?;
                self.optimizer.step(network, &gradients);
                total += value * indices.len() as Scalar;
            }
            history.push(total / data.sample_count() as Scalar);
        }
        Ok(history)
    }

    fn check_output_layer(&self, network: &Network) -> Result<(), String> {
        let activation = network.output_layer().activation();
        if self.loss.needs_raw_scores() && activation != Activation::Identity {
            return Err(format!(
                "this loss needs raw scores, so the output layer must be Identity, not {activation:?}"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::Dense;
    use crate::initializations::XavierNormal;
    use crate::losses::{MeanSquaredError, SoftmaxCrossEntropy};
    use crate::matrix::Matrix;
    use crate::network::Gradients;
    use crate::optimizers::Sgd;
    use std::collections::BTreeMap;

    struct Frozen;

    impl Optimizer for Frozen {
        fn name(&self) -> &'static str {
            "frozen"
        }

        fn settings(&self) -> BTreeMap<String, Scalar> {
            BTreeMap::new()
        }

        fn step(&mut self, _: &mut Network, _: &Gradients) {}
    }

    fn network(output: Activation) -> Network {
        let mut rng = Rng::new(1);
        Network::new(vec![
            Dense::random(2, 3, Activation::Tanh, &XavierNormal, &mut rng),
            Dense::random(3, 2, output, &XavierNormal, &mut rng),
        ])
        .unwrap()
    }

    fn data() -> Dataset {
        let inputs = Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.2]]).unwrap();
        let targets = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]).unwrap();
        Dataset::new(inputs, targets).unwrap()
    }

    fn trainer(loss: impl Loss + 'static, batch_size: usize, epochs: usize) -> Trainer {
        Trainer::new(loss, Sgd::new(0.1).unwrap(), batch_size, epochs).unwrap()
    }

    fn trained(seed: u64) -> Network {
        let mut network = network(Activation::Sigmoid);
        trainer(MeanSquaredError, 1, 20)
            .fit(&mut network, &data(), &mut Rng::new(seed))
            .unwrap();
        network
    }

    #[test]
    fn rejects_empty_batches_and_zero_epochs() {
        let sgd = Sgd::new(0.1).unwrap();
        assert_eq!(
            Trainer::new(MeanSquaredError, sgd, 0, 10).err(),
            Some("the batch size must be at least 1".to_string())
        );
        assert_eq!(
            Trainer::new(MeanSquaredError, sgd, 8, 0).err(),
            Some("training needs at least one epoch".to_string())
        );
        let trainer = Trainer::new(MeanSquaredError, sgd, 8, 10).unwrap();
        assert_eq!((trainer.batch_size(), trainer.epochs()), (8, 10));
    }

    #[test]
    fn softmax_cross_entropy_needs_an_identity_output_layer() {
        let mut network = network(Activation::Sigmoid);
        assert_eq!(
            trainer(SoftmaxCrossEntropy, 2, 1).fit(&mut network, &data(), &mut Rng::new(2)),
            Err(
                "this loss needs raw scores, so the output layer must be Identity, not Sigmoid"
                    .to_string()
            )
        );
    }

    #[test]
    fn returns_one_decreasing_loss_per_epoch() {
        let mut network = network(Activation::Identity);
        let history = trainer(SoftmaxCrossEntropy, 2, 200)
            .fit(&mut network, &data(), &mut Rng::new(2))
            .unwrap();
        assert_eq!(history.len(), 200);
        assert!(
            history[199] < history[0] / 2.0,
            "{} then {}",
            history[0],
            history[199]
        );
    }

    #[test]
    fn the_same_seed_trains_the_same_network() {
        assert_eq!(trained(3), trained(3));
    }

    #[test]
    fn the_seed_changes_the_order_of_the_samples() {
        assert_ne!(trained(3), trained(4));
    }

    #[test]
    fn accepts_a_batch_larger_than_the_data() {
        let mut network = network(Activation::Sigmoid);
        let history = trainer(MeanSquaredError, 100, 3)
            .fit(&mut network, &data(), &mut Rng::new(4))
            .unwrap();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn weights_each_batch_loss_by_its_size() {
        let mut network = network(Activation::Sigmoid);
        let data = data();
        let (whole, _) = network
            .gradients(data.inputs(), data.targets(), &MeanSquaredError)
            .unwrap();
        let history = Trainer::new(MeanSquaredError, Frozen, 3, 1)
            .unwrap()
            .fit(&mut network, &data, &mut Rng::new(6))
            .unwrap();
        assert!(
            (history[0] - whole).abs() < 1e-6,
            "{} instead of {whole}",
            history[0]
        );
    }

    #[test]
    fn reports_data_that_does_not_fit_the_network() {
        let mut network = network(Activation::Sigmoid);
        let data = Dataset::new(Matrix::zeros(2, 5), Matrix::zeros(2, 2)).unwrap();
        assert_eq!(
            trainer(MeanSquaredError, 1, 1).fit(&mut network, &data, &mut Rng::new(5)),
            Err("the network expects 2 inputs, got 5".to_string())
        );
    }
}

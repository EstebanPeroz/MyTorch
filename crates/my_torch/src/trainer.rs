use rand::seq::SliceRandom;

use crate::activations::Activation;
use crate::dataset::Dataset;
use crate::early_stopping::{EarlyStopping, Watch};
use crate::epoch_report::EpochReport;
use crate::history::History;
use crate::losses::Loss;
use crate::matrix::Scalar;
use crate::network::Network;
use crate::optimizers::Optimizer;
use crate::rng::Rng;

/// Rows per thread when computing the gradients of a batch.
const SHARD_ROWS: usize = 32;

pub struct Trainer {
    loss: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
    batch_size: usize,
    epochs: usize,
    early_stopping: Option<EarlyStopping>,
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
            early_stopping: None,
        })
    }

    pub fn with_early_stopping(mut self, early_stopping: EarlyStopping) -> Trainer {
        self.early_stopping = Some(early_stopping);
        self
    }

    pub fn loss(&self) -> &dyn Loss {
        self.loss.as_ref()
    }

    pub fn optimizer(&self) -> &dyn Optimizer {
        self.optimizer.as_ref()
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn early_stopping(&self) -> Option<EarlyStopping> {
        self.early_stopping
    }

    pub fn fit(
        &mut self,
        network: &mut Network,
        data: &Dataset,
        rng: &mut Rng,
    ) -> Result<History, String> {
        self.fit_observed(network, data, rng, |_| {})
    }

    /// Like `fit`, and calls `observer` at the end of every epoch, after the validation loss is measured
    /// and before early stopping decides whether to continue. The observer only reads: it cannot change the
    /// training, so the result is the same as `fit`'s.
    pub fn fit_observed(
        &mut self,
        network: &mut Network,
        data: &Dataset,
        rng: &mut Rng,
        mut observer: impl FnMut(&EpochReport),
    ) -> Result<History, String> {
        self.check_output_layer(network)?;
        let (mut order, validation_rows) = match self.early_stopping {
            Some(early_stopping) => match data.groups() {
                Some(groups) => early_stopping.split_groups(groups, rng)?,
                None => early_stopping.split(data.sample_count(), rng)?,
            },
            None => ((0..data.sample_count()).collect(), Vec::new()),
        };
        let validation = self.early_stopping.map(|_| data.select(&validation_rows));
        let mut watch = self
            .early_stopping
            .map(|early_stopping| Watch::new(early_stopping.patience()));
        let mut history = History::default();
        for epoch in 1..=self.epochs {
            let training = self.run_epoch(network, data, &mut order, rng)?;
            let validation = validation
                .as_ref()
                .map(|set| network.evaluate(set.inputs(), set.targets(), self.loss.as_ref()))
                .transpose()?;
            history.record(training, validation);
            observer(&EpochReport::new(
                epoch,
                self.epochs,
                (training, validation),
                network,
                (&order, &validation_rows),
            ));
            if let (Some(watch), Some(loss)) = (&mut watch, validation)
                && watch.should_stop(loss, network)
            {
                break;
            }
        }
        if let Some(best) = watch.and_then(Watch::best_network) {
            *network = best;
        }
        Ok(history)
    }

    fn run_epoch(
        &mut self,
        network: &mut Network,
        data: &Dataset,
        order: &mut [usize],
        rng: &mut Rng,
    ) -> Result<Scalar, String> {
        order.shuffle(rng.generator());
        let mut total = 0.0;
        for indices in order.chunks(self.batch_size) {
            let batch = data.select(indices);
            let (value, gradients) = network.gradients_sharded(
                batch.inputs(),
                batch.targets(),
                self.loss.as_ref(),
                SHARD_ROWS,
            )?;
            self.optimizer.step(network, &gradients);
            total += value * indices.len() as Scalar;
        }
        Ok(total / order.len() as Scalar)
    }

    pub(crate) fn check_output_layer(&self, network: &Network) -> Result<(), String> {
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
    use rand::RngExt;
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
        assert_eq!(trainer.loss().name(), "mean_squared_error");
        assert_eq!(trainer.optimizer().settings(), sgd.settings());
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
        let losses = history.training();
        assert_eq!(losses.len(), 200);
        assert!(
            losses[199] < losses[0] / 2.0,
            "{} then {}",
            losses[0],
            losses[199]
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
        assert_eq!(history.epochs(), 3);
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
            (history.training()[0] - whole).abs() < 1e-6,
            "{} instead of {whole}",
            history.training()[0]
        );
    }

    fn noise(samples: usize, seed: u64) -> Dataset {
        let mut rng = Rng::new(seed);
        let inputs = (0..samples * 3)
            .map(|_| rng.generator().random::<Scalar>())
            .collect();
        let mut targets = Matrix::zeros(samples, 2);
        for row in 0..samples {
            targets.set(row, rng.generator().random_range(0..2), 1.0);
        }
        Dataset::new(Matrix::from_vec(samples, 3, inputs).unwrap(), targets).unwrap()
    }

    fn wide_network() -> Network {
        let mut rng = Rng::new(7);
        Network::new(vec![
            Dense::random(3, 32, Activation::Tanh, &XavierNormal, &mut rng),
            Dense::random(32, 2, Activation::Identity, &XavierNormal, &mut rng),
        ])
        .unwrap()
    }

    #[test]
    fn without_early_stopping_every_epoch_runs_and_nothing_is_set_aside() {
        let mut trainer = trainer(SoftmaxCrossEntropy, 4, 30);
        assert_eq!(trainer.early_stopping(), None);
        let mut network = wide_network();
        let history = trainer
            .fit(&mut network, &noise(40, 8), &mut Rng::new(9))
            .unwrap();
        assert_eq!(history.epochs(), 30);
        assert!(history.validation().is_empty());
    }

    #[test]
    fn stops_when_the_validation_loss_stops_improving_and_keeps_the_best_network() {
        let early_stopping = EarlyStopping::new(0.25, 5).unwrap();
        let data = noise(40, 8);
        let mut network = wide_network();
        let mut trainer = Trainer::new(SoftmaxCrossEntropy, Sgd::new(0.5).unwrap(), 4, 500)
            .unwrap()
            .with_early_stopping(early_stopping);
        assert_eq!(trainer.early_stopping(), Some(early_stopping));
        let history = trainer.fit(&mut network, &data, &mut Rng::new(9)).unwrap();
        let best = history.best_epoch().unwrap();
        assert!(history.epochs() < 500, "{} epochs", history.epochs());
        assert_eq!(history.epochs(), best + 5 + 1);
        assert_eq!(history.validation().len(), history.epochs());
        let (_, validation) = early_stopping.split(40, &mut Rng::new(9)).unwrap();
        let set = data.select(&validation);
        assert_eq!(
            network.evaluate(set.inputs(), set.targets(), &SoftmaxCrossEntropy),
            Ok(history.validation()[best])
        );
    }

    #[test]
    fn an_observer_sees_every_epoch_and_changes_nothing() {
        let early_stopping = EarlyStopping::new(0.25, 5).unwrap();
        let data = noise(40, 8);
        let make = || {
            Trainer::new(SoftmaxCrossEntropy, Sgd::new(0.5).unwrap(), 4, 500)
                .unwrap()
                .with_early_stopping(early_stopping)
        };
        let mut plain = wide_network();
        let expected = make().fit(&mut plain, &data, &mut Rng::new(9)).unwrap();

        let mut observed = wide_network();
        let mut seen = Vec::new();
        let history = make()
            .fit_observed(&mut observed, &data, &mut Rng::new(9), |report| {
                let mut rows = [report.training_samples(), report.validation_samples()].concat();
                rows.sort_unstable();
                assert_eq!(rows, (0..40).collect::<Vec<_>>());
                assert_eq!(report.validation_samples().len(), 10);
                assert_eq!(report.epochs(), 500);
                seen.push((
                    report.epoch(),
                    report.training_loss(),
                    report.validation_loss(),
                ));
            })
            .unwrap();

        assert_eq!(history, expected);
        assert_eq!(observed, plain);
        assert_eq!(seen.len(), history.epochs());
        for (index, (epoch, training, validation)) in seen.into_iter().enumerate() {
            assert_eq!(epoch, index + 1);
            assert_eq!(training, history.training()[index]);
            assert_eq!(validation, Some(history.validation()[index]));
        }
    }

    #[test]
    fn grouped_samples_are_never_split_between_training_and_validation() {
        // 20 samples in 10 groups: sample `row` is in group `row / 2`.
        let groups: Vec<usize> = (0..20).map(|row| row / 2).collect();
        let data = noise(20, 8).with_groups(groups.clone()).unwrap();
        let mut network = wide_network();
        let mut calls = 0;
        trainer(SoftmaxCrossEntropy, 4, 3)
            .with_early_stopping(EarlyStopping::new(0.25, 5).unwrap())
            .fit_observed(&mut network, &data, &mut Rng::new(9), |report| {
                calls += 1;
                let validation: Vec<usize> = report
                    .validation_samples()
                    .iter()
                    .map(|&row| groups[row])
                    .collect();
                assert!(validation.len() >= 5);
                assert!(
                    report
                        .training_samples()
                        .iter()
                        .all(|&row| !validation.contains(&groups[row]))
                );
            })
            .unwrap();
        assert_eq!(calls, 3);
    }

    #[test]
    fn without_early_stopping_the_observer_gets_every_sample_for_training() {
        let mut network = wide_network();
        let mut calls = 0;
        trainer(SoftmaxCrossEntropy, 4, 3)
            .fit_observed(&mut network, &noise(12, 8), &mut Rng::new(9), |report| {
                calls += 1;
                assert_eq!(report.training_samples().len(), 12);
                assert!(report.validation_samples().is_empty());
                assert_eq!(report.validation_loss(), None);
                assert_eq!(report.network().input_size(), 3);
            })
            .unwrap();
        assert_eq!(calls, 3);
    }

    #[test]
    fn the_training_loss_only_counts_the_training_samples() {
        let early_stopping = EarlyStopping::new(0.25, 1).unwrap();
        let data = noise(8, 10);
        let mut network = wide_network();
        let history = Trainer::new(SoftmaxCrossEntropy, Frozen, 3, 1)
            .unwrap()
            .with_early_stopping(early_stopping)
            .fit(&mut network, &data, &mut Rng::new(11))
            .unwrap();
        let (training, _) = early_stopping.split(8, &mut Rng::new(11)).unwrap();
        let set = data.select(&training);
        let expected = network
            .evaluate(set.inputs(), set.targets(), &SoftmaxCrossEntropy)
            .unwrap();
        assert!(
            (history.training()[0] - expected).abs() < 1e-6,
            "{} instead of {expected}",
            history.training()[0]
        );
    }

    #[test]
    fn reports_too_few_samples_for_validation() {
        let mut network = wide_network();
        let data = noise(4, 12);
        assert_eq!(
            trainer(SoftmaxCrossEntropy, 2, 5)
                .with_early_stopping(EarlyStopping::new(0.1, 2).unwrap())
                .fit(&mut network, &data, &mut Rng::new(13))
                .err(),
            Some("4 samples are too few to set aside 0.1 of them for validation".to_string())
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

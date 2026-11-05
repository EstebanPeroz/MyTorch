use rand::seq::SliceRandom;

use crate::matrix::Scalar;
use crate::network::Network;
use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyStopping {
    validation: Scalar,
    patience: usize,
}

pub(crate) struct Watch {
    patience: usize,
    best: Option<(Scalar, Network)>,
    waited: usize,
}

impl EarlyStopping {
    pub fn new(validation: Scalar, patience: usize) -> Result<EarlyStopping, String> {
        if !(validation > 0.0 && validation < 1.0) {
            return Err(format!(
                "the validation share must be between 0 and 1, got {validation}"
            ));
        }
        if patience == 0 {
            return Err("the patience must be at least 1 epoch".into());
        }
        Ok(EarlyStopping {
            validation,
            patience,
        })
    }

    pub fn validation(&self) -> Scalar {
        self.validation
    }

    pub fn patience(&self) -> usize {
        self.patience
    }

    pub(crate) fn split(
        &self,
        count: usize,
        rng: &mut Rng,
    ) -> Result<(Vec<usize>, Vec<usize>), String> {
        let validation = (count as Scalar * self.validation).round() as usize;
        if validation == 0 || validation == count {
            return Err(format!(
                "{count} samples are too few to set aside {} of them for validation",
                self.validation
            ));
        }
        let mut order: Vec<usize> = (0..count).collect();
        order.shuffle(rng.generator());
        let training = order.split_off(validation);
        Ok((training, order))
    }
}

impl Watch {
    pub(crate) fn new(patience: usize) -> Watch {
        Watch {
            patience,
            best: None,
            waited: 0,
        }
    }

    pub(crate) fn should_stop(&mut self, loss: Scalar, network: &Network) -> bool {
        let improved = match &self.best {
            None => !loss.is_nan(),
            Some((best, _)) => loss < *best,
        };
        if improved {
            self.best = Some((loss, network.clone()));
            self.waited = 0;
            return false;
        }
        self.waited += 1;
        self.waited >= self.patience
    }

    pub(crate) fn best_network(self) -> Option<Network> {
        self.best.map(|(_, network)| network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::dense::Dense;
    use crate::matrix::Matrix;

    fn network(weight: Scalar) -> Network {
        let layer = Dense::new(
            Matrix::from_rows(&[[weight]]).unwrap(),
            Matrix::zeros(1, 1),
            Activation::Identity,
        )
        .unwrap();
        Network::new(vec![layer]).unwrap()
    }

    fn watch(patience: usize, losses: &[Scalar]) -> (Vec<bool>, Option<Network>) {
        let mut watch = Watch::new(patience);
        let stops = losses
            .iter()
            .enumerate()
            .map(|(epoch, &loss)| watch.should_stop(loss, &network(epoch as Scalar)))
            .collect();
        (stops, watch.best_network())
    }

    #[test]
    fn refuses_settings_that_make_no_sense() {
        for validation in [0.0, 1.0, -0.1, 1.5, Scalar::NAN] {
            assert_eq!(
                EarlyStopping::new(validation, 3),
                Err(format!(
                    "the validation share must be between 0 and 1, got {validation}"
                ))
            );
        }
        assert_eq!(
            EarlyStopping::new(0.2, 0),
            Err("the patience must be at least 1 epoch".to_string())
        );
        let early_stopping = EarlyStopping::new(0.2, 3).unwrap();
        assert_eq!(
            (early_stopping.validation(), early_stopping.patience()),
            (0.2, 3)
        );
    }

    #[test]
    fn splits_the_samples_at_random_without_losing_any() {
        let early_stopping = EarlyStopping::new(0.2, 3).unwrap();
        let (training, validation) = early_stopping.split(10, &mut Rng::new(1)).unwrap();
        assert_eq!((training.len(), validation.len()), (8, 2));
        let mut all: Vec<usize> = training.iter().chain(&validation).copied().collect();
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        assert_eq!(
            early_stopping.split(10, &mut Rng::new(1)),
            Ok((training, validation.clone()))
        );
        let (_, other) = early_stopping.split(10, &mut Rng::new(2)).unwrap();
        assert_ne!(other, validation);
    }

    #[test]
    fn refuses_to_split_too_few_samples() {
        assert_eq!(
            EarlyStopping::new(0.1, 3)
                .unwrap()
                .split(4, &mut Rng::new(1)),
            Err("4 samples are too few to set aside 0.1 of them for validation".to_string())
        );
        assert_eq!(
            EarlyStopping::new(0.9, 3)
                .unwrap()
                .split(2, &mut Rng::new(1)),
            Err("2 samples are too few to set aside 0.9 of them for validation".to_string())
        );
    }

    #[test]
    fn stops_after_patience_epochs_without_improvement_and_keeps_the_best() {
        let (stops, best) = watch(3, &[1.0, 0.8, 0.9, 0.85, 0.95]);
        assert_eq!(stops, [false, false, false, false, true]);
        assert_eq!(best, Some(network(1.0)));
    }

    #[test]
    fn an_improvement_starts_the_wait_again() {
        let (stops, best) = watch(2, &[1.0, 1.1, 0.9, 1.0, 0.9]);
        assert_eq!(stops, [false, false, false, false, true]);
        assert_eq!(best, Some(network(2.0)));
    }

    #[test]
    fn a_loss_that_is_not_a_number_is_never_an_improvement() {
        let (stops, best) = watch(2, &[1.0, Scalar::NAN, Scalar::NAN]);
        assert_eq!(stops, [false, false, true]);
        assert_eq!(best, Some(network(0.0)));
        let (_, best) = watch(2, &[Scalar::NAN, 1.0]);
        assert_eq!(best, Some(network(1.0)));
    }
}

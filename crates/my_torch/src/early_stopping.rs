use std::collections::HashMap;

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
                "the validation share must be greater than 0 and less than 1, got {validation}"
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

impl EarlyStopping {
    /// Like `split`, but samples with the same group number always end up on the same side. The validation
    /// set takes whole groups until it holds at least its share of the samples, so it can be a little larger.
    pub(crate) fn split_groups(
        &self,
        groups: &[usize],
        rng: &mut Rng,
    ) -> Result<(Vec<usize>, Vec<usize>), String> {
        let count = groups.len();
        let wanted = (count as Scalar * self.validation).round() as usize;
        let mut members: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut position = HashMap::new();
        for (row, &group) in groups.iter().enumerate() {
            let index = *position.entry(group).or_insert_with(|| {
                members.push((group, Vec::new()));
                members.len() - 1
            });
            members[index].1.push(row);
        }
        members.shuffle(rng.generator());
        let mut validation = Vec::new();
        let mut taken = 0;
        while validation.len() < wanted.max(1) && taken < members.len() {
            validation.extend(&members[taken].1);
            taken += 1;
        }
        let training: Vec<usize> = members[taken..]
            .iter()
            .flat_map(|(_, rows)| rows.iter().copied())
            .collect();
        if wanted == 0 || training.is_empty() {
            return Err(format!(
                "{count} samples in {} groups are too few to set aside {} of them for validation",
                members.len(),
                self.validation
            ));
        }
        Ok((training, validation))
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
    use std::collections::HashSet;

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
                    "the validation share must be greater than 0 and less than 1, got {validation}"
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
    fn a_split_by_groups_never_separates_a_group() {
        let early_stopping = EarlyStopping::new(0.25, 3).unwrap();
        // 10 groups of 2 samples: sample `row` is in group `row % 10`.
        let groups: Vec<usize> = (0..20).map(|row| row % 10).collect();
        let (training, validation) = early_stopping
            .split_groups(&groups, &mut Rng::new(1))
            .unwrap();
        // 25% of 20 samples is 5: three whole groups, which is 6 samples.
        assert_eq!((training.len(), validation.len()), (14, 6));
        let groups_of =
            |rows: &[usize]| -> HashSet<usize> { rows.iter().map(|&r| groups[r]).collect() };
        assert!(groups_of(&training).is_disjoint(&groups_of(&validation)));
        let mut all: Vec<usize> = training.iter().chain(&validation).copied().collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<_>>());
        assert_eq!(
            early_stopping.split_groups(&groups, &mut Rng::new(1)),
            Ok((training, validation.clone()))
        );
        let (_, other) = early_stopping
            .split_groups(&groups, &mut Rng::new(2))
            .unwrap();
        assert_ne!(other, validation);
    }

    #[test]
    fn refuses_to_split_too_few_groups() {
        let early_stopping = EarlyStopping::new(0.5, 3).unwrap();
        assert_eq!(
            early_stopping.split_groups(&[0, 0, 0, 0], &mut Rng::new(1)),
            Err(
                "4 samples in 1 groups are too few to set aside 0.5 of them for validation"
                    .to_string()
            )
        );
        assert!(
            EarlyStopping::new(0.1, 3)
                .unwrap()
                .split_groups(&[0, 1, 2, 3], &mut Rng::new(1))
                .is_err()
        );
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

use crate::matrix::Scalar;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    training: Vec<Scalar>,
    validation: Vec<Scalar>,
}

impl History {
    pub fn training(&self) -> &[Scalar] {
        &self.training
    }

    pub fn validation(&self) -> &[Scalar] {
        &self.validation
    }

    pub fn epochs(&self) -> usize {
        self.training.len()
    }

    pub fn best_epoch(&self) -> Option<usize> {
        self.validation
            .iter()
            .enumerate()
            .filter(|(_, loss)| !loss.is_nan())
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(epoch, _)| epoch)
    }

    pub(crate) fn record(&mut self, training: Scalar, validation: Option<Scalar>) {
        self.training.push(training);
        self.validation.extend(validation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_one_training_loss_per_epoch_in_order() {
        let mut history = History::default();
        assert_eq!(history.epochs(), 0);
        history.record(0.5, None);
        history.record(0.25, None);
        assert_eq!(history.training(), [0.5, 0.25]);
        assert_eq!(history.epochs(), 2);
        assert!(history.validation().is_empty());
        assert_eq!(history.best_epoch(), None);
    }

    #[test]
    fn the_best_epoch_has_the_lowest_validation_loss() {
        let mut history = History::default();
        for (training, validation) in [(0.9, 0.8), (0.7, Scalar::NAN), (0.5, 0.6), (0.3, 0.6)] {
            history.record(training, Some(validation));
        }
        assert_eq!(history.validation().len(), 4);
        assert_eq!(history.best_epoch(), Some(2));
    }
}

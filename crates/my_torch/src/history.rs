use crate::matrix::Scalar;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    training: Vec<Scalar>,
}

impl History {
    pub fn training(&self) -> &[Scalar] {
        &self.training
    }

    pub fn epochs(&self) -> usize {
        self.training.len()
    }

    pub(crate) fn record(&mut self, training: Scalar) {
        self.training.push(training);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_one_training_loss_per_epoch_in_order() {
        let mut history = History::default();
        assert_eq!(history.epochs(), 0);
        history.record(0.5);
        history.record(0.25);
        assert_eq!(history.training(), [0.5, 0.25]);
        assert_eq!(history.epochs(), 2);
    }
}

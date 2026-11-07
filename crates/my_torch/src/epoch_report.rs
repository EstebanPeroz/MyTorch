use crate::matrix::Scalar;
use crate::network::Network;

/// What a training run shows an observer at the end of each epoch.
///
/// The report borrows the network as it is after the epoch, and the row indices of the training and of the
/// validation samples in the dataset given to `Trainer::fit_observed`. That lets the caller measure anything
/// it needs (such as an accuracy) on its own data, without the library knowing the problem.
#[derive(Debug, Clone, Copy)]
pub struct EpochReport<'a> {
    epoch: usize,
    epochs: usize,
    training_loss: Scalar,
    validation_loss: Option<Scalar>,
    network: &'a Network,
    training: &'a [usize],
    validation: &'a [usize],
}

impl<'a> EpochReport<'a> {
    pub(crate) fn new(
        epoch: usize,
        epochs: usize,
        losses: (Scalar, Option<Scalar>),
        network: &'a Network,
        samples: (&'a [usize], &'a [usize]),
    ) -> EpochReport<'a> {
        EpochReport {
            epoch,
            epochs,
            training_loss: losses.0,
            validation_loss: losses.1,
            network,
            training: samples.0,
            validation: samples.1,
        }
    }

    /// The number of this epoch, starting at 1.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// The most epochs the training may run. Early stopping can end it sooner.
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn training_loss(&self) -> Scalar {
        self.training_loss
    }

    /// `None` when the training has no validation split.
    pub fn validation_loss(&self) -> Option<Scalar> {
        self.validation_loss
    }

    pub fn network(&self) -> &Network {
        self.network
    }

    /// The rows of the dataset used to train, in no particular order.
    pub fn training_samples(&self) -> &[usize] {
        self.training
    }

    /// The rows of the dataset held out for validation. Empty without a validation split.
    pub fn validation_samples(&self) -> &[usize] {
        self.validation
    }
}

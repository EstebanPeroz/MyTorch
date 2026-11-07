use std::collections::BTreeMap;

use serde::Deserialize;

use crate::early_stopping::EarlyStopping;
use crate::losses::loss_from_name;
use crate::matrix::Scalar;
use crate::optimizers::optimizer_from_settings;
use crate::trainer::Trainer;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainingFile {
    loss: String,
    optimizer: OptimizerFile,
    batch_size: usize,
    epochs: usize,
    seed: u64,
    #[serde(default)]
    early_stopping: Option<EarlyStoppingFile>,
}

#[derive(Deserialize)]
struct OptimizerFile {
    name: String,
    #[serde(flatten)]
    settings: BTreeMap<String, Scalar>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EarlyStoppingFile {
    validation: Scalar,
    patience: usize,
}

impl TrainingFile {
    pub(crate) fn from_parts(
        loss: String,
        optimizer: String,
        settings: BTreeMap<String, Scalar>,
        batch_size: usize,
        epochs: usize,
        seed: u64,
        early_stopping: Option<(Scalar, usize)>,
    ) -> TrainingFile {
        TrainingFile {
            loss,
            optimizer: OptimizerFile {
                name: optimizer,
                settings,
            },
            batch_size,
            epochs,
            seed,
            early_stopping: early_stopping.map(|(validation, patience)| EarlyStoppingFile {
                validation,
                patience,
            }),
        }
    }

    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    pub(crate) fn trainer(&self) -> Result<Trainer, String> {
        let trainer = Trainer::new(
            loss_from_name(&self.loss)?,
            optimizer_from_settings(&self.optimizer.name, &self.optimizer.settings)?,
            self.batch_size,
            self.epochs,
        )?;
        Ok(match &self.early_stopping {
            Some(early_stopping) => trainer.with_early_stopping(EarlyStopping::new(
                early_stopping.validation,
                early_stopping.patience,
            )?),
            None => trainer,
        })
    }
}

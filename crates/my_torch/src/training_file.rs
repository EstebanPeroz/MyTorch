use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::early_stopping::EarlyStopping;
use crate::losses::loss_from_name;
use crate::matrix::Scalar;
use crate::optimizers::optimizer_from_settings;
use crate::trainer::Trainer;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainingFile {
    loss: String,
    optimizer: OptimizerFile,
    batch_size: usize,
    epochs: usize,
    seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    early_stopping: Option<EarlyStoppingFile>,
}

#[derive(Serialize, Deserialize)]
struct OptimizerFile {
    name: String,
    #[serde(flatten)]
    settings: BTreeMap<String, Scalar>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EarlyStoppingFile {
    validation: Scalar,
    patience: usize,
}

impl TrainingFile {
    pub(crate) fn new(trainer: &Trainer, seed: u64) -> TrainingFile {
        TrainingFile {
            loss: trainer.loss().name().to_string(),
            optimizer: OptimizerFile {
                name: trainer.optimizer().name().to_string(),
                settings: trainer.optimizer().settings(),
            },
            batch_size: trainer.batch_size(),
            epochs: trainer.epochs(),
            seed,
            early_stopping: trainer
                .early_stopping()
                .map(|early_stopping| EarlyStoppingFile {
                    validation: early_stopping.validation(),
                    patience: early_stopping.patience(),
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

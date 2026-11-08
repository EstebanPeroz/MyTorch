use std::path::Path;

use my_torch::{Dataset, ModelConfig};
use my_torch_analyzer::chess_file::{ChessFile, Labels};
use my_torch_analyzer::encoding;
use my_torch_analyzer::sample::Sample;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde_json::{Value, json};

use crate::search::Score;
use crate::space::{Params, SearchSpace};

/// The chessboards of a search: a part to train on and a part, never trained on, to score the trials.
pub struct Data {
    training: Dataset,
    validation: Dataset,
    validation_samples: Vec<Sample>,
}

impl Data {
    /// Reads the chessboards, removes the duplicates, keeps at most `limit` of them picked at random, and holds
    /// out the share `validation` of them (at least one chessboard).
    pub fn read(
        path: &Path,
        limit: Option<usize>,
        validation: f64,
        seed: u64,
    ) -> Result<Data, String> {
        let file = ChessFile::read(path, Labels::Required)?;
        Data::from_samples(
            encoding::unique(file.samples().to_vec()),
            limit,
            validation,
            seed,
        )
    }

    fn from_samples(
        mut samples: Vec<Sample>,
        limit: Option<usize>,
        validation: f64,
        seed: u64,
    ) -> Result<Data, String> {
        samples.shuffle(&mut StdRng::seed_from_u64(seed));
        samples.truncate(limit.unwrap_or(usize::MAX));
        let held_out = ((samples.len() as f64 * validation).round() as usize).max(1);
        if held_out >= samples.len() {
            return Err(format!(
                "{} chessboards are not enough to hold out a validation split and train on the rest",
                samples.len()
            ));
        }
        let validation_samples = samples.split_off(samples.len() - held_out);
        Ok(Data {
            training: encoding::dataset(&samples)?,
            validation: encoding::dataset(&validation_samples)?,
            validation_samples,
        })
    }

    pub fn training_len(&self) -> usize {
        self.training.sample_count()
    }

    pub fn validation_len(&self) -> usize {
        self.validation_samples.len()
    }
}

/// Trains a new network for `epochs` epochs with the hyperparameters `params` on top of the `base`
/// configuration, and scores it on the validation split. The trial trains on exactly `epochs` epochs: the
/// early stopping of the base configuration is left out, because the validation split of the search replaces it.
pub fn evaluate(
    data: &Data,
    base: &Value,
    params: &Params,
    epochs: usize,
) -> Result<Score, String> {
    let config = trial_config(base, params, epochs)?;
    let mut model = ModelConfig::from_json(&config.to_string())?.generate(0);
    model.train(&data.training)?;
    let labels = encoding::predict(model.network(), &data.validation_samples)?;
    let correct = labels
        .iter()
        .zip(&data.validation_samples)
        .filter(|(label, sample)| Some(**label) == sample.label())
        .count();
    let loss = model.network().evaluate(
        data.validation.inputs(),
        data.validation.targets(),
        model.trainer().loss(),
    )?;
    if !loss.is_finite() {
        return Err("the training diverged".into());
    }
    Ok(Score {
        accuracy: correct as f64 / labels.len() as f64,
        loss: f64::from(loss),
    })
}

fn trial_config(base: &Value, params: &Params, epochs: usize) -> Result<Value, String> {
    let mut config = SearchSpace::apply(base, params)?;
    let training = config
        .get_mut("training")
        .and_then(Value::as_object_mut)
        .ok_or("the base configuration has no \"training\" section")?;
    training.insert("epochs".into(), json!(epochs));
    training.remove("early_stopping");
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"{
        "inputs": 769,
        "layers": [
            { "neurons": 8, "activation": "relu", "initialization": "he_normal" },
            { "neurons": 6, "activation": "identity", "initialization": "xavier_normal" }
        ],
        "training": {
            "loss": "softmax_cross_entropy",
            "optimizer": { "name": "sgd", "learning_rate": 0.05 },
            "batch_size": 4, "epochs": 100, "seed": 3,
            "early_stopping": { "validation": 0.1, "patience": 5 }
        }
    }"#;

    fn samples() -> Vec<Sample> {
        let file = ChessFile::read(
            Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/boards.txt"
            )),
            Labels::Required,
        )
        .unwrap();
        encoding::unique(file.samples().to_vec())
    }

    fn base() -> Value {
        serde_json::from_str(BASE).unwrap()
    }

    #[test]
    fn splits_the_chessboards_without_losing_any() {
        let total = samples().len();
        let data = Data::from_samples(samples(), None, 0.25, 1).unwrap();
        assert_eq!(data.training_len() + data.validation_len(), total);
        assert_eq!(
            data.validation_len(),
            (total as f64 * 0.25).round() as usize
        );
        assert!(
            data.validation_samples
                .iter()
                .all(|s| samples().contains(s))
        );
    }

    #[test]
    fn limits_the_chessboards_and_splits_them_the_same_way_for_a_seed() {
        let split = |seed| {
            let data = Data::from_samples(samples(), Some(8), 0.25, seed).unwrap();
            assert_eq!((data.training_len(), data.validation_len()), (6, 2));
            data.validation_samples
        };
        assert_eq!(split(1), split(1));
        assert_ne!(split(1), split(2));
    }

    #[test]
    fn refuses_too_few_chessboards() {
        let error = Data::from_samples(samples(), Some(1), 0.2, 1)
            .err()
            .unwrap();
        assert!(error.contains("1 chessboards are not enough"), "{error}");
    }

    #[test]
    fn a_trial_trains_for_the_given_epochs_without_early_stopping() {
        let config = trial_config(&base(), &Params::new(), 7).unwrap();
        assert_eq!(config["training"]["epochs"], 7);
        assert!(config["training"].get("early_stopping").is_none());
        assert_eq!(base()["training"]["epochs"], 100);
    }

    #[test]
    fn scores_a_network_between_zero_and_one() {
        let data = Data::from_samples(samples(), None, 0.25, 1).unwrap();
        let params = Params::from([("learning_rate".to_string(), json!(0.01))]);
        let score = evaluate(&data, &base(), &params, 2).unwrap();
        assert!((0.0..=1.0).contains(&score.accuracy), "{score:?}");
        assert!(score.loss.is_finite() && score.loss > 0.0, "{score:?}");
        assert_eq!(score, evaluate(&data, &base(), &params, 2).unwrap());
    }

    #[test]
    fn a_diverging_network_is_a_failed_trial() {
        let data = Data::from_samples(samples(), None, 0.25, 1).unwrap();
        let params = Params::from([("learning_rate".to_string(), json!(1e30))]);
        assert!(evaluate(&data, &base(), &params, 3).is_err());
    }
}

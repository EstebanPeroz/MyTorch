use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::activations::Activation;
use crate::dataset::Dataset;
use crate::dense::Dense;
use crate::history::History;
use crate::losses::loss_from_name;
use crate::matrix::{Matrix, Scalar};
use crate::network::Network;
use crate::optimizers::optimizer_from_settings;
use crate::rng::Rng;
use crate::trainer::Trainer;

const FORMAT_VERSION: u32 = 1;

pub struct Model {
    network: Network,
    trainer: Trainer,
    seed: u64,
}

#[derive(Deserialize)]
struct FileVersion {
    version: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelFile {
    version: u32,
    layers: Vec<LayerFile>,
    training: TrainingFile,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerFile {
    activation: String,
    weights: Vec<Vec<Scalar>>,
    biases: Vec<Scalar>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingFile {
    loss: String,
    optimizer: OptimizerFile,
    batch_size: usize,
    epochs: usize,
    seed: u64,
}

#[derive(Serialize, Deserialize)]
struct OptimizerFile {
    name: String,
    #[serde(flatten)]
    settings: BTreeMap<String, Scalar>,
}

impl Model {
    pub fn new(network: Network, trainer: Trainer, seed: u64) -> Result<Model, String> {
        trainer.check_output_layer(&network)?;
        Ok(Model {
            network,
            trainer,
            seed,
        })
    }

    pub fn network(&self) -> &Network {
        &self.network
    }

    pub fn trainer(&self) -> &Trainer {
        &self.trainer
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn train(&mut self, data: &Dataset) -> Result<History, String> {
        self.trainer
            .fit(&mut self.network, data, &mut Rng::new(self.seed))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let json = self.to_json()?;
        let temporary = temporary_path(path);
        fs::write(&temporary, json + "\n")
            .and_then(|()| fs::rename(&temporary, path))
            .map_err(|error| {
                let _ = fs::remove_file(&temporary);
                format!("cannot write {}: {error}", path.display())
            })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Model, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        Model::from_json(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn to_json(&self) -> Result<String, String> {
        let layers = self
            .network
            .layers()
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                LayerFile::from_dense(layer)
                    .map_err(|error| format!("layer {}: {error}", index + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let file = ModelFile {
            version: FORMAT_VERSION,
            layers,
            training: TrainingFile {
                loss: self.trainer.loss().name().to_string(),
                optimizer: OptimizerFile {
                    name: self.trainer.optimizer().name().to_string(),
                    settings: self.trainer.optimizer().settings(),
                },
                batch_size: self.trainer.batch_size(),
                epochs: self.trainer.epochs(),
                seed: self.seed,
            },
        };
        Ok(serde_json::to_string(&file).expect("a model file only holds names and finite numbers"))
    }

    pub fn from_json(text: &str) -> Result<Model, String> {
        let FileVersion { version } = serde_json::from_str(text).map_err(invalid_file)?;
        if version != FORMAT_VERSION {
            return Err(format!(
                "unsupported network file version {version}, expected {FORMAT_VERSION}"
            ));
        }
        let file: ModelFile = serde_json::from_str(text).map_err(invalid_file)?;
        let layers = file
            .layers
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                layer
                    .into_dense()
                    .map_err(|error| format!("layer {}: {error}", index + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let training = file.training;
        let trainer = Trainer::new(
            loss_from_name(&training.loss)?,
            optimizer_from_settings(&training.optimizer.name, &training.optimizer.settings)?,
            training.batch_size,
            training.epochs,
        )?;
        Model::new(Network::new(layers)?, trainer, training.seed)
    }
}

impl LayerFile {
    fn from_dense(layer: &Dense) -> Result<LayerFile, String> {
        check_finite(layer).map_err(|error| format!("{error}, so the network cannot be saved"))?;
        let weights = layer.weights();
        Ok(LayerFile {
            activation: layer.activation().name().to_string(),
            weights: (0..weights.rows())
                .map(|row| weights.row(row).to_vec())
                .collect(),
            biases: layer.biases().data().to_vec(),
        })
    }

    fn into_dense(self) -> Result<Dense, String> {
        let activation = Activation::from_name(&self.activation)?;
        let weights = Matrix::from_rows(&self.weights)?;
        let biases = Matrix::from_rows(&[self.biases])?;
        let layer = Dense::new(weights, biases, activation)?;
        check_finite(&layer)?;
        Ok(layer)
    }
}

fn check_finite(layer: &Dense) -> Result<(), String> {
    let mut parameters = layer.weights().data().iter().chain(layer.biases().data());
    if parameters.any(|value| !value.is_finite()) {
        return Err("a weight or bias is not a finite number".into());
    }
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

fn invalid_file(error: serde_json::Error) -> String {
    format!("invalid network file: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::architecture::{Architecture, LayerSpec};
    use crate::dense::Dense;
    use crate::initializations::XavierNormal;
    use crate::losses::{MeanSquaredError, SoftmaxCrossEntropy};
    use crate::optimizers::Sgd;

    fn network(output: Activation) -> Network {
        Architecture::new(
            2,
            vec![
                LayerSpec::new(3, Activation::Tanh, XavierNormal).unwrap(),
                LayerSpec::new(2, output, XavierNormal).unwrap(),
            ],
        )
        .unwrap()
        .build(&mut Rng::new(1))
    }

    fn trainer() -> Trainer {
        Trainer::new(SoftmaxCrossEntropy, Sgd::new(0.1).unwrap(), 2, 5).unwrap()
    }

    fn data() -> Dataset {
        Dataset::new(
            Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).unwrap(),
            Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]).unwrap(),
        )
        .unwrap()
    }

    const SMALL: &str = r#"{"version":1,"layers":[{"activation":"identity","weights":[[0.5],[-1.0]],"biases":[0.25]}],"training":{"loss":"mean_squared_error","optimizer":{"name":"sgd","learning_rate":0.1},"batch_size":4,"epochs":10,"seed":7}}"#;

    fn small() -> Model {
        let layer = Dense::new(
            Matrix::from_rows(&[[0.5], [-1.0]]).unwrap(),
            Matrix::from_rows(&[[0.25]]).unwrap(),
            Activation::Identity,
        )
        .unwrap();
        let trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1).unwrap(), 4, 10).unwrap();
        Model::new(Network::new(vec![layer]).unwrap(), trainer, 7).unwrap()
    }

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("my_torch_{name}_{}.nn", std::process::id()))
    }

    #[test]
    fn writes_the_network_and_its_training_settings_as_json() {
        assert_eq!(small().to_json(), Ok(SMALL.to_string()));
    }

    #[test]
    fn reading_back_gives_the_same_model() {
        let model = Model::new(network(Activation::Identity), trainer(), 3).unwrap();
        let json = model.to_json().unwrap();
        let loaded = Model::from_json(&json).unwrap();
        assert_eq!(loaded.network(), model.network());
        assert_eq!(loaded.seed(), 3);
        assert_eq!(loaded.trainer().loss().name(), "softmax_cross_entropy");
        assert_eq!(
            loaded.trainer().optimizer().settings(),
            model.trainer().optimizer().settings()
        );
        assert_eq!(
            (loaded.trainer().batch_size(), loaded.trainer().epochs()),
            (2, 5)
        );
        assert_eq!(loaded.to_json(), Ok(json));
    }

    #[test]
    fn keeps_every_weight_exactly() {
        let weights = [[0.1, 1.0 / 3.0], [Scalar::MAX, -1e-45]];
        let layer = Dense::new(
            Matrix::from_rows(&weights).unwrap(),
            Matrix::from_rows(&[[Scalar::MIN_POSITIVE, -7.25e-12]]).unwrap(),
            Activation::Relu,
        )
        .unwrap();
        let trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1).unwrap(), 4, 10).unwrap();
        let model = Model::new(Network::new(vec![layer]).unwrap(), trainer, 7).unwrap();
        let loaded = Model::from_json(&model.to_json().unwrap()).unwrap();
        assert_eq!(loaded.network(), model.network());
    }

    #[test]
    fn refuses_to_save_weights_that_are_not_numbers() {
        let layer = Dense::new(
            Matrix::from_rows(&[[Scalar::NAN]]).unwrap(),
            Matrix::from_rows(&[[0.0]]).unwrap(),
            Activation::Identity,
        )
        .unwrap();
        let trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1).unwrap(), 4, 10).unwrap();
        let model = Model::new(Network::new(vec![layer]).unwrap(), trainer, 7).unwrap();
        let message =
            "layer 1: a weight or bias is not a finite number, so the network cannot be saved";
        assert_eq!(model.to_json(), Err(message.to_string()));
        let path = temporary("not_a_number");
        assert_eq!(model.save(&path), Err(message.to_string()));
        assert!(!path.exists());
    }

    #[test]
    fn refuses_broken_files() {
        let layer = r#"{"activation":"identity","weights":[[0.5],[-1.0]],"biases":[0.25]}"#;
        let changed = |old: &str, new: &str| {
            assert!(SMALL.contains(old), "{old}");
            SMALL.replace(old, new)
        };
        let cases = [
            (
                "not json".to_string(),
                "invalid network file: expected ident at line 1 column 2",
            ),
            (
                "{}".to_string(),
                "invalid network file: missing field `version` at line 1 column 2",
            ),
            (
                changed(r#""version":1"#, r#""version":2"#),
                "unsupported network file version 2, expected 1",
            ),
            (
                changed(r#""identity""#, r#""softmax""#),
                "layer 1: unknown activation \"softmax\", expected one of: identity, sigmoid, tanh, relu, leaky_relu",
            ),
            (
                changed("[[0.5],[-1.0]]", "[[0.5],[-1.0,2.0]]"),
                "layer 1: matrix row 2 has 2 values, expected 1",
            ),
            (
                changed(r#""weights":[[0.5],[-1.0]]"#, r#""weights":[]"#),
                "layer 1: a matrix cannot be 0 × 0",
            ),
            (
                changed("[[0.5],[-1.0]]", "[[1e39],[-1.0]]"),
                "layer 1: a weight or bias is not a finite number",
            ),
            (
                changed(r#""biases":[0.25]"#, r#""biases":[-1e39]"#),
                "layer 1: a weight or bias is not a finite number",
            ),
            (
                changed(r#""biases":[0.25]"#, r#""biases":[0.25,1.0]"#),
                "layer 1: a layer with 1 outputs needs 1 × 1 biases, got a 1 × 2 matrix",
            ),
            (changed(layer, ""), "a network needs at least one layer"),
            (
                changed(layer, &format!("{layer},{layer}")),
                "layer 1 has 1 outputs but layer 2 expects 2 inputs",
            ),
            (
                changed("mean_squared_error", "huber"),
                "unknown loss \"huber\", expected one of: mean_squared_error, softmax_cross_entropy",
            ),
            (
                changed(r#""name":"sgd""#, r#""name":"adam""#),
                "unknown optimizer \"adam\", expected one of: sgd",
            ),
            (
                changed(r#""learning_rate":0.1"#, r#""learnig_rate":0.1"#),
                "the sgd optimizer has no setting called learnig_rate",
            ),
            (
                changed(r#""learning_rate":0.1"#, r#""learning_rate":-1.0"#),
                "the learning rate must be a positive number, got -1",
            ),
            (
                changed(r#""batch_size":4"#, r#""batch_size":0"#),
                "the batch size must be at least 1",
            ),
            (
                changed("mean_squared_error", "softmax_cross_entropy")
                    .replace("identity", "sigmoid"),
                "this loss needs raw scores, so the output layer must be Identity, not Sigmoid",
            ),
            (
                changed(r#""seed":7"#, r#""seed":7,"extra":1"#),
                "invalid network file: unknown field `extra`",
            ),
            (
                changed(r#""seed":7"#, r#""seed":-7"#),
                "invalid network file: invalid value: integer `-7`",
            ),
        ];
        for (text, expected) in cases {
            let error = Model::from_json(&text).err().unwrap_or_default();
            assert!(
                error.starts_with(expected),
                "{text}\n gave: {error}\n expected: {expected}"
            );
        }
    }

    #[test]
    fn saves_and_loads_through_a_file() {
        let path = temporary("round_trip");
        small().save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let loaded = Model::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(text, format!("{SMALL}\n"));
        assert_eq!(loaded.unwrap().to_json(), Ok(SMALL.to_string()));
    }

    #[test]
    fn saving_replaces_the_previous_file() {
        let path = temporary("replaced");
        Model::new(network(Activation::Identity), trainer(), 3)
            .unwrap()
            .save(&path)
            .unwrap();
        small().save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(text, format!("{SMALL}\n"));
        assert!(!temporary_path(&path).exists());
    }

    #[test]
    fn a_failed_save_keeps_the_previous_file() {
        let path = temporary("kept");
        std::fs::write(&path, "previous network").unwrap();
        let blocked = temporary_path(&path);
        std::fs::create_dir(&blocked).unwrap();
        let error = small().save(&path).err().unwrap_or_default();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_dir(&blocked).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(
            error.starts_with(&format!("cannot write {}: ", path.display())),
            "{error}"
        );
        assert_eq!(text, "previous network");
    }

    #[test]
    fn a_failed_save_leaves_no_temporary_file() {
        let path = temporary("folder_in_the_way");
        std::fs::create_dir(&path).unwrap();
        let error = small().save(&path).err().unwrap_or_default();
        let leftover = temporary_path(&path).exists();
        std::fs::remove_dir(&path).unwrap();
        assert!(
            error.starts_with(&format!("cannot write {}: ", path.display())),
            "{error}"
        );
        assert!(!leftover);
    }

    #[test]
    fn reports_files_that_cannot_be_read_or_written() {
        let missing = temporary("missing");
        let error = Model::load(&missing).err().unwrap_or_default();
        assert!(
            error.starts_with(&format!("cannot read {}: ", missing.display())),
            "{error}"
        );
        let broken = temporary("broken");
        std::fs::write(&broken, "{}").unwrap();
        let error = Model::load(&broken).err();
        std::fs::remove_file(&broken).unwrap();
        assert_eq!(
            error,
            Some(format!(
                "{}: invalid network file: missing field `version` at line 1 column 2",
                broken.display()
            ))
        );
        let nowhere = temporary("no_such_folder").join("model.nn");
        let error = small().save(&nowhere).err().unwrap_or_default();
        assert!(
            error.starts_with(&format!("cannot write {}: ", nowhere.display())),
            "{error}"
        );
    }

    #[test]
    fn refuses_a_loss_that_does_not_fit_the_output_layer() {
        assert_eq!(
            Model::new(network(Activation::Sigmoid), trainer(), 3).err(),
            Some(
                "this loss needs raw scores, so the output layer must be Identity, not Sigmoid"
                    .to_string()
            )
        );
        let mse = Trainer::new(MeanSquaredError, Sgd::new(0.1).unwrap(), 2, 5).unwrap();
        assert!(Model::new(network(Activation::Sigmoid), mse, 3).is_ok());
    }

    #[test]
    fn keeps_its_network_trainer_and_seed() {
        let model = Model::new(network(Activation::Identity), trainer(), 3).unwrap();
        assert_eq!(model.network(), &network(Activation::Identity));
        assert_eq!(model.trainer().loss().name(), "softmax_cross_entropy");
        assert_eq!(model.trainer().optimizer().name(), "sgd");
        assert_eq!(
            (model.trainer().batch_size(), model.trainer().epochs()),
            (2, 5)
        );
        assert_eq!(model.seed(), 3);
    }

    #[test]
    fn trains_its_network_with_its_own_seed() {
        let mut model = Model::new(network(Activation::Identity), trainer(), 3).unwrap();
        let history = model.train(&data()).unwrap();
        let mut expected = network(Activation::Identity);
        let expected_history = trainer()
            .fit(&mut expected, &data(), &mut Rng::new(3))
            .unwrap();
        assert_eq!(history, expected_history);
        assert_eq!(model.network(), &expected);
        assert_ne!(model.network(), &network(Activation::Identity));
    }
}

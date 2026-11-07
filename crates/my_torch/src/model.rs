use std::fs;
use std::path::{Path, PathBuf};

use crate::dataset::Dataset;
use crate::dense::Dense;
use crate::epoch_report::EpochReport;
use crate::history::History;
use crate::matrix::Scalar;
use crate::model_file;
use crate::network::Network;
use crate::rng::Rng;
use crate::trainer::Trainer;

pub struct Model {
    network: Network,
    trainer: Trainer,
    seed: u64,
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

    /// Like `train`, and calls `observer` at the end of every epoch. See `Trainer::fit_observed`.
    pub fn train_observed(
        &mut self,
        data: &Dataset,
        observer: impl FnMut(&EpochReport),
    ) -> Result<History, String> {
        self.trainer
            .fit_observed(&mut self.network, data, &mut Rng::new(self.seed), observer)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        let temporary = temporary_path(path);
        fs::write(&temporary, bytes)
            .and_then(|()| fs::rename(&temporary, path))
            .map_err(|error| {
                let _ = fs::remove_file(&temporary);
                format!("cannot write {}: {error}", path.display())
            })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Model, String> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        Model::from_bytes(&bytes)
            .map_err(|error| format!("{}: invalid network file: {error}", path.display()))
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        model_file::encode(self.network.layers(), &self.trainer, self.seed)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        model_file::decode(bytes)
    }

    /// A readable summary of the model, for `--dump`: the file it makes, each layer and the training settings.
    pub fn describe(&self) -> Result<String, String> {
        let size = self.to_bytes()?.len();
        let layers = self.network.layers();
        let mut shape = vec![self.network.input_size()];
        shape.extend(layers.iter().map(Dense::output_size));
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        let parameters: usize = layers
            .iter()
            .map(|layer| layer.weights().data().len() + layer.biases().data().len())
            .sum();
        let mut text = format!(
            "file: {size} bytes, format version {}\nnetwork: {} ({} layers, {parameters} parameters)\n",
            model_file::FORMAT_VERSION,
            shape.join(" → "),
            layers.len(),
        );
        for (index, layer) in layers.iter().enumerate() {
            let weights = layer.weights().data();
            let min = weights.iter().copied().fold(Scalar::INFINITY, Scalar::min);
            let max = weights
                .iter()
                .copied()
                .fold(Scalar::NEG_INFINITY, Scalar::max);
            let mean = weights.iter().sum::<Scalar>() / weights.len() as Scalar;
            text += &format!(
                "  layer {}: {} → {}, {}, {} weights (min {min:.4}, max {max:.4}, mean {mean:.4}), {} biases\n",
                index + 1,
                layer.input_size(),
                layer.output_size(),
                layer.activation().name(),
                weights.len(),
                layer.biases().data().len(),
            );
            if layer.dropout() > 0.0 {
                text.pop();
                text += &format!(", dropout {}\n", layer.dropout());
            }
        }
        let optimizer = self.trainer.optimizer();
        let settings: Vec<String> = optimizer
            .settings()
            .iter()
            .map(|(name, value)| format!("{name} {value}"))
            .collect();
        text += &format!(
            "training:\n  loss: {}\n  optimizer: {} ({})\n  batch size: {}\n  epochs: {}\n  seed: {}\n",
            self.trainer.loss().name(),
            optimizer.name(),
            settings.join(", "),
            self.trainer.batch_size(),
            self.trainer.epochs(),
            self.seed,
        );
        text += &match self.trainer.early_stopping() {
            Some(early_stopping) => format!(
                "  early stopping: validation share {}, patience {}\n",
                early_stopping.validation(),
                early_stopping.patience()
            ),
            None => "  early stopping: off\n".to_string(),
        };
        Ok(text)
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::architecture::{Architecture, LayerSpec};
    use crate::dense::Dense;
    use crate::early_stopping::EarlyStopping;
    use crate::initializations::XavierNormal;
    use crate::losses::{MeanSquaredError, SoftmaxCrossEntropy};
    use crate::matrix::Matrix;
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

    /// What `small()` must be saved as, written by hand.
    fn small_bytes() -> Vec<u8> {
        let mut bytes = b"MTNN".to_vec();
        bytes.extend(1u32.to_le_bytes()); // version
        bytes.extend(1u32.to_le_bytes()); // layers
        bytes.extend(b"\x08identity");
        bytes.extend(2u32.to_le_bytes()); // inputs
        bytes.extend(1u32.to_le_bytes()); // neurons
        for value in [0.5f32, -1.0, 0.25] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(b"\x12mean_squared_error");
        bytes.extend(b"\x03sgd");
        bytes.extend(1u32.to_le_bytes()); // optimizer settings
        bytes.extend(b"\x0dlearning_rate");
        bytes.extend(0.1f32.to_le_bytes());
        for value in [4u64, 10, 7] {
            bytes.extend(value.to_le_bytes()); // batch size, epochs, seed
        }
        bytes.push(0); // no early stopping
        bytes
    }

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("my_torch_{name}_{}.nn", std::process::id()))
    }

    #[test]
    fn writes_the_network_and_its_training_settings_as_bytes() {
        assert_eq!(small().to_bytes(), Ok(small_bytes()));
        assert_eq!(small_bytes().len(), 111);
    }

    #[test]
    fn reading_back_gives_the_same_model() {
        let model = Model::new(network(Activation::Identity), trainer(), 3).unwrap();
        let bytes = model.to_bytes().unwrap();
        let loaded = Model::from_bytes(&bytes).unwrap();
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
        assert_eq!(loaded.to_bytes(), Ok(bytes));
    }

    #[test]
    fn keeps_the_early_stopping_settings() {
        let early_stopping = EarlyStopping::new(0.1, 5).unwrap();
        let model = Model::new(
            network(Activation::Identity),
            trainer().with_early_stopping(early_stopping),
            3,
        )
        .unwrap();
        let loaded = Model::from_bytes(&model.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.trainer().early_stopping(), Some(early_stopping));
        let plain = Model::from_bytes(&small_bytes()).unwrap();
        assert_eq!(plain.trainer().early_stopping(), None);
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
        let loaded = Model::from_bytes(&model.to_bytes().unwrap()).unwrap();
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
        assert_eq!(model.to_bytes(), Err(message.to_string()));
        let path = temporary("not_a_number");
        assert_eq!(model.save(&path), Err(message.to_string()));
        assert!(!path.exists());
    }

    #[test]
    fn refuses_broken_files() {
        let valid = small_bytes();
        let replace = |base: &[u8], old: &[u8], new: &[u8]| {
            let at = base
                .windows(old.len())
                .enumerate()
                .filter(|(_, window)| *window == old)
                .map(|(at, _)| at)
                .collect::<Vec<_>>();
            assert_eq!(at.len(), 1, "{old:?} must appear once");
            [&base[..at[0]], new, &base[at[0] + old.len()..]].concat()
        };
        let changed = |old: &[u8], new: &[u8]| replace(&valid, old, new);
        let early_stopping = |validation: f32, patience: u64| {
            let mut bytes = valid[..valid.len() - 1].to_vec();
            bytes.push(1);
            bytes.extend(validation.to_le_bytes());
            bytes.extend(patience.to_le_bytes());
            bytes
        };
        let batch_and_epochs = [4u64.to_le_bytes(), 10u64.to_le_bytes()].concat();
        let sizes = [2u32.to_le_bytes(), 1u32.to_le_bytes()].concat();
        let cases: Vec<(Vec<u8>, &str)> = vec![
            (
                b"{}".to_vec(),
                "the file ends too early while reading the header (4 bytes needed, 2 left)",
            ),
            (
                b"not a network".to_vec(),
                "not a network file (wrong header)",
            ),
            (
                changed(b"MTNN\x01", b"MTNN\x02"),
                "unsupported network file version 2, expected 1",
            ),
            (
                valid[..30].to_vec(),
                "layer 1: the file ends too early while reading the weights (8 bytes needed, 1 left)",
            ),
            (
                changed(
                    &sizes,
                    &[u32::MAX.to_le_bytes(), 1u32.to_le_bytes()].concat(),
                ),
                "layer 1: the file ends too early while reading the weights",
            ),
            (
                changed(&sizes, &[2u32.to_le_bytes(), 0u32.to_le_bytes()].concat()),
                "layer 1: a matrix cannot be 2 × 0",
            ),
            (
                changed(b"\x08identity", b"\x07softmax"),
                "layer 1: unknown activation \"softmax\", expected one of: identity, sigmoid, tanh, relu, leaky_relu",
            ),
            (
                changed(&0.5f32.to_le_bytes(), &f32::NAN.to_le_bytes()),
                "layer 1: a weight or bias is not a finite number",
            ),
            (
                changed(&0.25f32.to_le_bytes(), &f32::INFINITY.to_le_bytes()),
                "layer 1: a weight or bias is not a finite number",
            ),
            (
                changed(b"\x12mean_squared_error", b"\x05huber"),
                "unknown loss \"huber\", expected one of: mean_squared_error, softmax_cross_entropy",
            ),
            (
                changed(b"\x03sgd", b"\x04adam"),
                "unknown optimizer \"adam\", expected one of: sgd",
            ),
            (
                changed(b"\x0dlearning_rate", b"\x0dlearnin_grate"),
                "the sgd optimizer has no setting called learnin_grate",
            ),
            (
                changed(&0.1f32.to_le_bytes(), &(-1.0f32).to_le_bytes()),
                "the learning rate must be a positive number, got -1",
            ),
            (
                changed(
                    &batch_and_epochs,
                    &[0u64.to_le_bytes(), 10u64.to_le_bytes()].concat(),
                ),
                "the batch size must be at least 1",
            ),
            (
                replace(
                    &changed(b"\x08identity", b"\x07sigmoid"),
                    b"\x12mean_squared_error",
                    b"\x15softmax_cross_entropy",
                ),
                "this loss needs raw scores, so the output layer must be Identity, not Sigmoid",
            ),
            (
                [&valid[..valid.len() - 1], &[2u8][..]].concat(),
                "invalid early stopping flag 2, expected 0 or 1",
            ),
            (
                early_stopping(1.5, 5),
                "the validation share must be greater than 0 and less than 1, got 1.5",
            ),
            (
                early_stopping(0.1, 0),
                "the patience must be at least 1 epoch",
            ),
            (
                [&valid[..], &[0u8][..]].concat(),
                "1 unexpected bytes after the end of the network",
            ),
            (
                valid[..valid.len() - 1].to_vec(),
                "the file ends too early while reading the early stopping flag",
            ),
        ];
        for (bytes, expected) in cases {
            let error = Model::from_bytes(&bytes).err().unwrap_or_default();
            assert!(
                error.starts_with(expected),
                "{bytes:?}\n gave: {error}\n expected: {expected}"
            );
        }
    }

    #[test]
    fn refuses_layers_that_do_not_fit_together() {
        let layer = |inputs, outputs| {
            Dense::new(
                Matrix::zeros(inputs, outputs),
                Matrix::zeros(1, outputs),
                Activation::Identity,
            )
            .unwrap()
        };
        let trainer = Trainer::new(MeanSquaredError, Sgd::new(0.1).unwrap(), 4, 10).unwrap();
        let encode = |layers: &[Dense]| model_file::encode(layers, &trainer, 7).unwrap();
        assert_eq!(
            Model::from_bytes(&encode(&[layer(2, 1), layer(2, 1)])).err(),
            Some("layer 1 has 1 outputs but layer 2 expects 2 inputs".to_string())
        );
        assert_eq!(
            Model::from_bytes(&encode(&[])).err(),
            Some("a network needs at least one layer".to_string())
        );
    }

    #[test]
    fn saves_and_loads_through_a_file() {
        let path = temporary("round_trip");
        small().save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let loaded = Model::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bytes, small_bytes());
        assert_eq!(loaded.unwrap().to_bytes(), Ok(small_bytes()));
    }

    #[test]
    fn saving_replaces_the_previous_file() {
        let path = temporary("replaced");
        Model::new(network(Activation::Identity), trainer(), 3)
            .unwrap()
            .save(&path)
            .unwrap();
        small().save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bytes, small_bytes());
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
                "{}: invalid network file: the file ends too early while reading the header (4 bytes needed, 2 left)",
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
    fn describes_the_model() {
        let expected = "file: 111 bytes, format version 1\n\
network: 2 → 1 (1 layers, 3 parameters)\n  \
layer 1: 2 → 1, identity, 2 weights (min -1.0000, max 0.5000, mean -0.2500), 1 biases\n\
training:\n  loss: mean_squared_error\n  optimizer: sgd (learning_rate 0.1)\n  batch size: 4\n  \
epochs: 10\n  seed: 7\n  early stopping: off\n";
        assert_eq!(small().describe(), Ok(expected.to_string()));
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

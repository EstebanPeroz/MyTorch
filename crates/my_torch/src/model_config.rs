use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::activations::Activation;
use crate::architecture::{Architecture, LayerSpec};
use crate::initializations::initialization_from_name;
use crate::model::Model;
use crate::rng::Rng;
use crate::training_file::TrainingFile;

pub struct ModelConfig {
    architecture: Architecture,
    training: TrainingFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    inputs: usize,
    layers: Vec<LayerConfigFile>,
    training: TrainingFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerConfigFile {
    neurons: usize,
    activation: String,
    initialization: String,
}

impl ModelConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<ModelConfig, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        ModelConfig::from_json(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn from_json(text: &str) -> Result<ModelConfig, String> {
        let file: ConfigFile = serde_json::from_str(text)
            .map_err(|error| format!("invalid configuration file: {error}"))?;
        let layers = file
            .layers
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                layer
                    .into_spec()
                    .map_err(|error| format!("layer {}: {error}", index + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let config = ModelConfig {
            architecture: Architecture::new(file.inputs, layers)?,
            training: file.training,
        };
        config.try_generate(0)?;
        Ok(config)
    }

    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }

    pub fn seed(&self) -> u64 {
        self.training.seed()
    }

    pub fn generate(&self, index: u64) -> Model {
        self.try_generate(index)
            .expect("the configuration was checked when it was read")
    }

    fn try_generate(&self, index: u64) -> Result<Model, String> {
        let seed = self.seed().wrapping_add(index);
        let network = self.architecture.build(&mut Rng::new(seed));
        Model::new(network, self.training.trainer()?, seed)
    }
}

impl LayerConfigFile {
    fn into_spec(self) -> Result<LayerSpec, String> {
        LayerSpec::new(
            self.neurons,
            Activation::from_name(&self.activation)?,
            initialization_from_name(&self.initialization)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const CONFIG: &str = r#"{
        "inputs": 2,
        "layers": [
            { "neurons": 3, "activation": "tanh", "initialization": "xavier_normal" },
            { "neurons": 2, "activation": "identity", "initialization": "he_normal" }
        ],
        "training": {
            "loss": "softmax_cross_entropy",
            "optimizer": { "name": "sgd", "learning_rate": 0.1 },
            "batch_size": 4,
            "epochs": 10,
            "seed": 7,
            "early_stopping": { "validation": 0.25, "patience": 3 }
        }
    }"#;

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "my_torch_config_{name}_{}.conf",
            std::process::id()
        ))
    }

    #[test]
    fn reads_the_architecture_and_the_training_settings() {
        let config = ModelConfig::from_json(CONFIG).unwrap();
        let architecture = config.architecture();
        assert_eq!(architecture.inputs(), 2);
        let layers: Vec<_> = architecture
            .layers()
            .iter()
            .map(|layer| {
                (
                    layer.neurons(),
                    layer.activation().name(),
                    layer.initialization().name(),
                )
            })
            .collect();
        assert_eq!(
            layers,
            [(3, "tanh", "xavier_normal"), (2, "identity", "he_normal")]
        );
        assert_eq!(config.seed(), 7);
    }

    #[test]
    fn each_network_gets_its_own_seed() {
        let config = ModelConfig::from_json(CONFIG).unwrap();
        let first = config.generate(0);
        let second = config.generate(1);
        assert_eq!((first.seed(), second.seed()), (7, 8));
        assert_eq!(
            second.network(),
            &config.architecture().build(&mut Rng::new(8))
        );
        assert_ne!(first.network(), second.network());
        assert_eq!(config.generate(1).network(), second.network());
    }

    #[test]
    fn a_generated_network_holds_the_training_section_of_its_configuration() {
        let model = ModelConfig::from_json(CONFIG).unwrap().generate(2);
        let loaded = Model::from_bytes(&model.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.trainer().loss().name(), "softmax_cross_entropy");
        assert_eq!(
            loaded.trainer().optimizer().settings(),
            BTreeMap::from([("learning_rate".to_string(), 0.1)])
        );
        assert_eq!(
            (loaded.trainer().batch_size(), loaded.trainer().epochs()),
            (4, 10)
        );
        assert_eq!(loaded.seed(), 9);
        let early_stopping = loaded.trainer().early_stopping().unwrap();
        assert_eq!(
            (early_stopping.validation(), early_stopping.patience()),
            (0.25, 3)
        );
    }

    #[test]
    fn refuses_broken_configurations() {
        let changed = |old: &str, new: &str| {
            assert!(CONFIG.contains(old), "{old}");
            CONFIG.replace(old, new)
        };
        let cases = [
            (
                "not json".to_string(),
                "invalid configuration file: expected ident at line 1 column 2",
            ),
            (
                "{}".to_string(),
                "invalid configuration file: missing field `inputs` at line 1 column 2",
            ),
            (
                changed(r#""inputs": 2,"#, r#""inputs": 2, "outputs": 2,"#),
                "invalid configuration file: unknown field `outputs`",
            ),
            (
                changed(r#""inputs": 2"#, r#""inputs": 0"#),
                "a network needs at least one input",
            ),
            (
                changed(r#""neurons": 2,"#, r#""neurons": 0,"#),
                "layer 2: a layer needs at least one neuron",
            ),
            (
                changed(r#""tanh""#, r#""softmax""#),
                "layer 1: unknown activation \"softmax\", expected one of: identity, sigmoid, tanh, relu, leaky_relu",
            ),
            (
                changed(r#""he_normal""#, r#""uniform""#),
                "layer 2: unknown initialization \"uniform\", expected one of: xavier_normal, he_normal",
            ),
            (
                changed(r#""loss": "softmax_cross_entropy""#, r#""loss": "huber""#),
                "unknown loss \"huber\", expected one of: mean_squared_error, softmax_cross_entropy",
            ),
            (
                changed(r#""learning_rate": 0.1"#, r#""learning_rate": 0"#),
                "the learning rate must be a positive number, got 0",
            ),
            (
                changed(r#""batch_size": 4"#, r#""batch_size": 0"#),
                "the batch size must be at least 1",
            ),
            (
                changed(r#""validation": 0.25"#, r#""validation": 1"#),
                "the validation share must be greater than 0 and less than 1, got 1",
            ),
            (
                changed(r#""identity""#, r#""sigmoid""#),
                "this loss needs raw scores, so the output layer must be Identity, not Sigmoid",
            ),
        ];
        for (text, expected) in cases {
            let error = ModelConfig::from_json(&text).err().unwrap_or_default();
            assert!(
                error.starts_with(expected),
                "{text}\n gave: {error}\n expected: {expected}"
            );
        }
        let without_layers = r#"{"inputs": 2, "layers": [], "training": {"loss": "mean_squared_error", "optimizer": {"name": "sgd", "learning_rate": 0.1}, "batch_size": 4, "epochs": 10, "seed": 7}}"#;
        assert_eq!(
            ModelConfig::from_json(without_layers).err(),
            Some("a network needs at least one layer".to_string())
        );
    }

    #[test]
    fn loads_a_file_and_names_it_in_errors() {
        let valid = temporary("valid");
        std::fs::write(&valid, CONFIG).unwrap();
        let loaded = ModelConfig::load(&valid);
        std::fs::remove_file(&valid).unwrap();
        assert_eq!(loaded.unwrap().seed(), 7);
        let broken = temporary("broken");
        std::fs::write(&broken, "{}").unwrap();
        let error = ModelConfig::load(&broken).err();
        std::fs::remove_file(&broken).unwrap();
        assert_eq!(
            error,
            Some(format!(
                "{}: invalid configuration file: missing field `inputs` at line 1 column 2",
                broken.display()
            ))
        );
        let missing = temporary("missing");
        let error = ModelConfig::load(&missing).err().unwrap_or_default();
        assert!(
            error.starts_with(&format!("cannot read {}: ", missing.display())),
            "{error}"
        );
    }
}

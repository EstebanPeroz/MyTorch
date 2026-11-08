use std::collections::BTreeMap;

use my_torch::Activation;
use rand::RngExt;
use rand::rngs::StdRng;
use serde::Deserialize;
use serde_json::{Value, json};

/// The values chosen for one trial, by hyperparameter name.
pub type Params = BTreeMap<String, Value>;

const SEARCHABLE: [&str; 5] = [
    "learning_rate",
    "batch_size",
    "dropout",
    "activation",
    "hidden",
];

/// What a hyperparameter can take: one of a list, or any number in a range.
#[derive(Debug, Clone, PartialEq)]
enum Dimension {
    Choices(Vec<Value>),
    Uniform(f64, f64),
    LogUniform(f64, f64),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DimensionFile {
    choices: Option<Vec<Value>>,
    uniform: Option<[f64; 2]>,
    log_uniform: Option<[f64; 2]>,
}

/// The hyperparameters to search and their ranges. Read from a JSON file such as:
///
/// ```json
/// { "learning_rate": { "log_uniform": [0.005, 0.2] }, "batch_size": { "choices": [64, 128] } }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSpace {
    dimensions: BTreeMap<String, Dimension>,
}

impl SearchSpace {
    pub fn from_json(text: &str) -> Result<SearchSpace, String> {
        let files: BTreeMap<String, DimensionFile> =
            serde_json::from_str(text).map_err(|error| format!("invalid search space: {error}"))?;
        if files.is_empty() {
            return Err("the search space is empty, it needs at least one hyperparameter".into());
        }
        let mut dimensions = BTreeMap::new();
        for (name, file) in files {
            if !SEARCHABLE.contains(&name.as_str()) {
                return Err(format!(
                    "unknown hyperparameter \"{name}\", expected one of: {}",
                    SEARCHABLE.join(", ")
                ));
            }
            let dimension = file
                .into_dimension()
                .and_then(|dimension| check(&name, &dimension).map(|()| dimension))
                .map_err(|error| format!("hyperparameter \"{name}\": {error}"))?;
            dimensions.insert(name, dimension);
        }
        Ok(SearchSpace { dimensions })
    }

    /// Draws one value for every hyperparameter.
    pub fn sample(&self, rng: &mut StdRng) -> Params {
        self.dimensions
            .iter()
            .map(|(name, dimension)| (name.clone(), dimension.sample(rng)))
            .collect()
    }

    /// The `base` configuration (a parsed `.conf`) with the hyperparameters of `params` applied. The hidden
    /// layers are every layer but the last one, which gives the outputs and is never touched.
    pub fn apply(base: &Value, params: &Params) -> Result<Value, String> {
        let mut config = base.clone();
        let layers = config
            .get_mut("layers")
            .and_then(Value::as_array_mut)
            .filter(|layers| layers.len() >= 2)
            .ok_or("the base configuration needs at least one hidden layer and an output layer")?;
        let output = layers.pop().expect("there are two layers");
        if let Some(widths) = params.get("hidden") {
            let template = layers[0].clone();
            *layers = widths
                .as_array()
                .expect("checked when the space was read")
                .iter()
                .map(|width| {
                    let mut layer = template.clone();
                    layer["neurons"] = width.clone();
                    layer
                })
                .collect();
        }
        for layer in layers.iter_mut() {
            if let Some(activation) = params.get("activation") {
                layer["activation"] = activation.clone();
                layer["initialization"] = json!(initialization(activation));
            }
            if let Some(dropout) = params.get("dropout") {
                layer["dropout"] = dropout.clone();
            }
        }
        layers.push(output);
        let training = config
            .get_mut("training")
            .and_then(Value::as_object_mut)
            .ok_or("the base configuration has no \"training\" section")?;
        if let Some(rate) = params.get("learning_rate") {
            let optimizer = training
                .get_mut("optimizer")
                .and_then(Value::as_object_mut)
                .ok_or("the base configuration has no \"optimizer\"")?;
            optimizer.insert("learning_rate".into(), rate.clone());
        }
        if let Some(size) = params.get("batch_size") {
            training.insert("batch_size".into(), size.clone());
        }
        Ok(config)
    }
}

/// He's initialization suits the rectifiers, Xavier's the others.
fn initialization(activation: &Value) -> &'static str {
    match activation.as_str() {
        Some("relu" | "leaky_relu") => "he_normal",
        _ => "xavier_normal",
    }
}

impl DimensionFile {
    fn into_dimension(self) -> Result<Dimension, String> {
        match (self.choices, self.uniform, self.log_uniform) {
            (Some(choices), None, None) => Ok(Dimension::Choices(choices)),
            (None, Some([low, high]), None) => Ok(Dimension::Uniform(low, high)),
            (None, None, Some([low, high])) => Ok(Dimension::LogUniform(low, high)),
            _ => Err("give exactly one of \"choices\", \"uniform\" or \"log_uniform\"".into()),
        }
    }
}

fn check(name: &str, dimension: &Dimension) -> Result<(), String> {
    match dimension {
        Dimension::Choices(choices) => {
            if choices.is_empty() {
                return Err("\"choices\" is empty".into());
            }
            choices
                .iter()
                .try_for_each(|choice| check_choice(name, choice))
        }
        Dimension::Uniform(low, high) | Dimension::LogUniform(low, high) => {
            if !matches!(name, "learning_rate" | "dropout") {
                return Err("only \"choices\" are possible for this hyperparameter".into());
            }
            if !(low.is_finite() && high.is_finite() && low < high) {
                return Err(format!("the range [{low}, {high}] needs low < high"));
            }
            if name == "dropout" && (*low < 0.0 || *high > 1.0) {
                return Err(format!(
                    "the range [{low}, {high}] is not valid, dropout rates are in [0, 1)"
                ));
            }
            if name == "learning_rate" && *low <= 0.0 {
                return Err(format!(
                    "the range [{low}, {high}] is not valid, a learning rate is positive"
                ));
            }
            Ok(())
        }
    }
}

fn check_choice(name: &str, choice: &Value) -> Result<(), String> {
    let (valid, expected) = match name {
        "learning_rate" => (
            choice.as_f64().is_some_and(|rate| rate > 0.0),
            "a positive number",
        ),
        "dropout" => (
            choice
                .as_f64()
                .is_some_and(|rate| (0.0..1.0).contains(&rate)),
            "a number in [0, 1)",
        ),
        "batch_size" => (
            choice.as_u64().is_some_and(|size| size > 0),
            "a whole number >= 1",
        ),
        "activation" => {
            return match choice.as_str() {
                Some(name) => Activation::from_name(name).map(|_| ()),
                None => Err(format!(
                    "{choice} is not valid, expected an activation name"
                )),
            };
        }
        _ => (
            choice.as_array().is_some_and(|widths| {
                !widths.is_empty() && widths.iter().all(|w| w.as_u64().is_some_and(|w| w > 0))
            }),
            "a non-empty list of layer widths >= 1",
        ),
    };
    if valid {
        Ok(())
    } else {
        Err(format!("{choice} is not valid, expected {expected}"))
    }
}

impl Dimension {
    fn sample(&self, rng: &mut StdRng) -> Value {
        match self {
            Dimension::Choices(choices) => choices[rng.random_range(0..choices.len())].clone(),
            Dimension::Uniform(low, high) => round(rng.random_range(*low..*high)),
            Dimension::LogUniform(low, high) => round(rng.random_range(low.ln()..high.ln()).exp()),
        }
    }
}

/// Four significant digits keep the logs and the written configuration readable.
fn round(value: f64) -> Value {
    if value == 0.0 {
        return json!(0.0);
    }
    let factor = 10f64.powi(3 - value.abs().log10().floor() as i32);
    json!((value * factor).round() / factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const SPACE: &str = r#"{
        "learning_rate": { "log_uniform": [0.001, 0.1] },
        "batch_size": { "choices": [32, 64] },
        "dropout": { "uniform": [0.0, 0.5] },
        "activation": { "choices": ["relu", "tanh"] },
        "hidden": { "choices": [[16, 8], [32]] }
    }"#;

    fn base() -> Value {
        serde_json::from_str(
            r#"{
            "inputs": 4,
            "layers": [
                { "neurons": 8, "activation": "relu", "initialization": "he_normal", "dropout": 0.2 },
                { "neurons": 3, "activation": "identity", "initialization": "xavier_normal" }
            ],
            "training": {
                "loss": "softmax_cross_entropy",
                "optimizer": { "name": "sgd", "learning_rate": 0.05 },
                "batch_size": 128, "epochs": 100, "seed": 42
            }
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn samples_stay_inside_the_ranges() {
        let space = SearchSpace::from_json(SPACE).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..200 {
            let params = space.sample(&mut rng);
            let rate = params["learning_rate"].as_f64().unwrap();
            assert!((0.001..=0.1).contains(&rate), "{rate}");
            let dropout = params["dropout"].as_f64().unwrap();
            assert!((0.0..=0.5).contains(&dropout), "{dropout}");
            assert!([32, 64].contains(&params["batch_size"].as_u64().unwrap()));
            assert!(["relu", "tanh"].contains(&params["activation"].as_str().unwrap()));
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_trials() {
        let space = SearchSpace::from_json(SPACE).unwrap();
        let draw = |seed| space.sample(&mut StdRng::seed_from_u64(seed));
        assert_eq!(draw(3), draw(3));
        assert_ne!(draw(3), draw(4));
    }

    #[test]
    fn log_uniform_covers_every_decade() {
        let space =
            SearchSpace::from_json(r#"{"learning_rate":{"log_uniform":[0.001,1.0]}}"#).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let rates: Vec<f64> = (0..300)
            .map(|_| space.sample(&mut rng)["learning_rate"].as_f64().unwrap())
            .collect();
        for decade in [(0.001, 0.01), (0.01, 0.1), (0.1, 1.0)] {
            let inside = rates.iter().filter(|r| (decade.0..=decade.1).contains(*r));
            assert!(inside.count() > 60, "{decade:?}");
        }
    }

    #[test]
    fn rounds_to_four_significant_digits() {
        assert_eq!(round(0.0123456), json!(0.01235));
        assert_eq!(round(123.456), json!(123.5));
        assert_eq!(round(0.0), json!(0.0));
    }

    #[test]
    fn applies_the_hyperparameters_to_the_base_configuration() {
        let params = Params::from([
            ("learning_rate".to_string(), json!(0.01)),
            ("batch_size".to_string(), json!(64)),
            ("dropout".to_string(), json!(0.3)),
            ("activation".to_string(), json!("tanh")),
            ("hidden".to_string(), json!([16, 8])),
        ]);
        let config = SearchSpace::apply(&base(), &params).unwrap();
        let layers = config["layers"].as_array().unwrap();
        let widths: Vec<_> = layers
            .iter()
            .map(|layer| layer["neurons"].as_u64().unwrap())
            .collect();
        assert_eq!(widths, [16, 8, 3]);
        assert_eq!(layers[1]["activation"], "tanh");
        assert_eq!(layers[1]["initialization"], "xavier_normal");
        assert_eq!(layers[1]["dropout"], 0.3);
        assert_eq!(layers[2]["activation"], "identity");
        assert!(layers[2].get("dropout").is_none());
        assert_eq!(config["training"]["optimizer"]["learning_rate"], 0.01);
        assert_eq!(config["training"]["batch_size"], 64);
        assert_eq!(config["training"]["epochs"], 100);
        my_torch::ModelConfig::from_json(&config.to_string()).unwrap();
    }

    #[test]
    fn leaves_the_rest_alone_without_hyperparameters() {
        assert_eq!(SearchSpace::apply(&base(), &Params::new()).unwrap(), base());
    }

    #[test]
    fn refuses_a_base_without_a_hidden_layer() {
        let mut config = base();
        config["layers"].as_array_mut().unwrap().remove(0);
        let error = SearchSpace::apply(&config, &Params::new()).unwrap_err();
        assert!(error.contains("at least one hidden layer"), "{error}");
    }

    #[test]
    fn refuses_broken_search_spaces() {
        let cases = [
            ("not json", "invalid search space: expected ident"),
            ("{}", "the search space is empty"),
            (
                r#"{"momentum":{"choices":[1]}}"#,
                "unknown hyperparameter \"momentum\", expected one of: learning_rate",
            ),
            (
                r#"{"dropout":{}}"#,
                "\"dropout\": give exactly one of \"choices\"",
            ),
            (
                r#"{"dropout":{"choices":[0.1],"uniform":[0,1]}}"#,
                "give exactly one of",
            ),
            (r#"{"dropout":{"choices":[]}}"#, "\"choices\" is empty"),
            (
                r#"{"dropout":{"choices":[1.0]}}"#,
                "1.0 is not valid, expected a number in [0, 1)",
            ),
            (
                r#"{"batch_size":{"choices":[0]}}"#,
                "0 is not valid, expected a whole number >= 1",
            ),
            (
                r#"{"batch_size":{"uniform":[1,2]}}"#,
                "only \"choices\" are possible",
            ),
            (
                r#"{"learning_rate":{"uniform":[0.2,0.1]}}"#,
                "needs low < high",
            ),
            (
                r#"{"learning_rate":{"log_uniform":[0.0,0.1]}}"#,
                "a learning rate is positive",
            ),
            (
                r#"{"dropout":{"uniform":[0.0,1.5]}}"#,
                "dropout rates are in [0, 1)",
            ),
            (
                r#"{"hidden":{"choices":[[]]}}"#,
                "expected a non-empty list of layer widths",
            ),
            (r#"{"dropout":{"step":1}}"#, "unknown field `step`"),
            (
                r#"{"activation":{"choices":["gelu"]}}"#,
                "unknown activation \"gelu\", expected one of",
            ),
            (
                r#"{"activation":{"choices":[1]}}"#,
                "1 is not valid, expected an activation name",
            ),
        ];
        for (text, message) in cases {
            let error = SearchSpace::from_json(text).unwrap_err();
            assert!(error.contains(message), "{text}: {error}");
        }
    }
}

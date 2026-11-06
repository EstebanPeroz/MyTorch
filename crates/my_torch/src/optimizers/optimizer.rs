use std::collections::BTreeMap;

use super::sgd::Sgd;
use crate::matrix::Scalar;
use crate::names::find_by_name;
use crate::network::{Gradients, Network};

pub trait Optimizer {
    fn name(&self) -> &'static str;

    fn settings(&self) -> BTreeMap<String, Scalar>;

    fn step(&mut self, network: &mut Network, gradients: &Gradients);
}

impl<O: Optimizer + ?Sized> Optimizer for Box<O> {
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    fn settings(&self) -> BTreeMap<String, Scalar> {
        self.as_ref().settings()
    }

    fn step(&mut self, network: &mut Network, gradients: &Gradients) {
        self.as_mut().step(network, gradients);
    }
}

type Constructor = fn(&BTreeMap<String, Scalar>) -> Result<Box<dyn Optimizer>, String>;

pub fn optimizer_from_settings(
    name: &str,
    settings: &BTreeMap<String, Scalar>,
) -> Result<Box<dyn Optimizer>, String> {
    let all: Vec<(&'static str, Constructor)> = vec![(Sgd::NAME, |settings| {
        Ok(Box::new(Sgd::from_settings(settings)?))
    })];
    let (_, constructor) = find_by_name("optimizer", name, all, |entry| entry.0)?;
    constructor(settings)
}

pub(super) fn read_settings(
    optimizer: &str,
    settings: &BTreeMap<String, Scalar>,
    keys: &[&str],
) -> Result<Vec<Scalar>, String> {
    if let Some(unknown) = settings.keys().find(|key| !keys.contains(&key.as_str())) {
        return Err(format!(
            "the {optimizer} optimizer has no setting called {unknown}"
        ));
    }
    keys.iter()
        .map(|key| {
            settings
                .get(*key)
                .copied()
                .ok_or_else(|| format!("the {optimizer} optimizer needs a {key} setting"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::dense::Dense;
    use crate::losses::MeanSquaredError;
    use crate::matrix::Matrix;

    fn settings(pairs: &[(&str, Scalar)]) -> BTreeMap<String, Scalar> {
        pairs
            .iter()
            .map(|&(key, value)| (key.to_string(), value))
            .collect()
    }

    #[test]
    fn an_optimizer_is_rebuilt_from_its_name_and_settings() {
        let sgd = Sgd::new(0.25).unwrap();
        let rebuilt = optimizer_from_settings(sgd.name(), &sgd.settings()).unwrap();
        assert_eq!(rebuilt.name(), "sgd");
        assert_eq!(rebuilt.settings(), settings(&[("learning_rate", 0.25)]));
    }

    #[test]
    fn a_boxed_optimizer_takes_the_same_step() {
        let layer = Dense::new(
            Matrix::from_rows(&[[1.0]]).unwrap(),
            Matrix::from_rows(&[[0.0]]).unwrap(),
            Activation::Identity,
        )
        .unwrap();
        let mut direct = Network::new(vec![layer]).unwrap();
        let mut boxed = direct.clone();
        let (_, gradients) = direct
            .gradients(
                &Matrix::from_rows(&[[2.0]]).unwrap(),
                &Matrix::from_rows(&[[0.0]]).unwrap(),
                &MeanSquaredError,
            )
            .unwrap();
        Sgd::new(0.125).unwrap().step(&mut direct, &gradients);
        optimizer_from_settings("sgd", &settings(&[("learning_rate", 0.125)]))
            .unwrap()
            .step(&mut boxed, &gradients);
        assert_eq!(boxed, direct);
        assert_ne!(direct.layers()[0].weights().get(0, 0), 1.0);
    }

    #[test]
    fn rejects_an_unknown_optimizer() {
        assert_eq!(
            optimizer_from_settings("adam", &settings(&[])).err(),
            Some("unknown optimizer \"adam\", expected one of: sgd".to_string())
        );
    }

    #[test]
    fn rejects_missing_and_unknown_settings() {
        assert_eq!(
            optimizer_from_settings("sgd", &settings(&[])).err(),
            Some("the sgd optimizer needs a learning_rate setting".to_string())
        );
        assert_eq!(
            optimizer_from_settings(
                "sgd",
                &settings(&[("learning_rate", 0.1), ("momentum", 0.9)])
            )
            .err(),
            Some("the sgd optimizer has no setting called momentum".to_string())
        );
        assert_eq!(
            optimizer_from_settings("sgd", &settings(&[("learning_rate", -1.0)])).err(),
            Some("the learning rate must be a positive number, got -1".to_string())
        );
    }
}

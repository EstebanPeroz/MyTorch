use crate::activations::Activation;
use crate::dense::Dense;
use crate::initializations::Initialization;
use crate::network::Network;
use crate::rng::Rng;

#[derive(Debug)]
pub struct Architecture {
    inputs: usize,
    layers: Vec<LayerSpec>,
}

#[derive(Debug)]
pub struct LayerSpec {
    neurons: usize,
    activation: Activation,
    initialization: Box<dyn Initialization>,
}

impl Architecture {
    pub fn new(inputs: usize, layers: Vec<LayerSpec>) -> Result<Architecture, String> {
        if inputs == 0 {
            return Err("a network needs at least one input".into());
        }
        if layers.is_empty() {
            return Err("a network needs at least one layer".into());
        }
        Ok(Architecture { inputs, layers })
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn layers(&self) -> &[LayerSpec] {
        &self.layers
    }

    pub fn build(&self, rng: &mut Rng) -> Network {
        let mut inputs = self.inputs;
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let dense = Dense::random(
                    inputs,
                    layer.neurons,
                    layer.activation,
                    layer.initialization.as_ref(),
                    rng,
                );
                inputs = layer.neurons;
                dense
            })
            .collect();
        Network::new(layers).expect("each layer takes the outputs of the one before it")
    }
}

impl LayerSpec {
    pub fn new(
        neurons: usize,
        activation: Activation,
        initialization: impl Initialization + 'static,
    ) -> Result<LayerSpec, String> {
        if neurons == 0 {
            return Err("a layer needs at least one neuron".into());
        }
        Ok(LayerSpec {
            neurons,
            activation,
            initialization: Box::new(initialization),
        })
    }

    pub fn neurons(&self) -> usize {
        self.neurons
    }

    pub fn activation(&self) -> Activation {
        self.activation
    }

    pub fn initialization(&self) -> &dyn Initialization {
        self.initialization.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initializations::{HeNormal, XavierNormal};

    fn architecture() -> Architecture {
        Architecture::new(
            3,
            vec![
                LayerSpec::new(4, Activation::Relu, HeNormal).unwrap(),
                LayerSpec::new(2, Activation::Identity, XavierNormal).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn rejects_a_network_without_inputs_or_layers() {
        let layer = || LayerSpec::new(2, Activation::Identity, XavierNormal).unwrap();
        assert_eq!(
            Architecture::new(0, vec![layer()]).err(),
            Some("a network needs at least one input".to_string())
        );
        assert_eq!(
            Architecture::new(3, vec![]).err(),
            Some("a network needs at least one layer".to_string())
        );
    }

    #[test]
    fn rejects_a_layer_without_neurons() {
        assert_eq!(
            LayerSpec::new(0, Activation::Relu, HeNormal).err(),
            Some("a layer needs at least one neuron".to_string())
        );
    }

    #[test]
    fn reads_back_the_description() {
        let architecture = architecture();
        assert_eq!(architecture.inputs(), 3);
        let [first, second] = architecture.layers() else {
            panic!("two layers");
        };
        assert_eq!((first.neurons(), first.activation()), (4, Activation::Relu));
        assert_eq!(
            (second.neurons(), second.activation()),
            (2, Activation::Identity)
        );
        let mut rng = Rng::new(1);
        assert_eq!(
            first.initialization().weights(4, 4, &mut rng),
            HeNormal.weights(4, 4, &mut Rng::new(1))
        );
    }

    #[test]
    fn builds_each_layer_from_the_outputs_of_the_one_before() {
        let mut rng = Rng::new(5);
        let expected = Network::new(vec![
            Dense::random(3, 4, Activation::Relu, &HeNormal, &mut rng),
            Dense::random(4, 2, Activation::Identity, &XavierNormal, &mut rng),
        ])
        .unwrap();
        assert_eq!(architecture().build(&mut Rng::new(5)), expected);
    }

    #[test]
    fn the_seed_decides_the_starting_weights() {
        let architecture = architecture();
        assert_eq!(
            architecture.build(&mut Rng::new(6)),
            architecture.build(&mut Rng::new(6))
        );
        assert_ne!(
            architecture.build(&mut Rng::new(6)),
            architecture.build(&mut Rng::new(7))
        );
    }
}

use crate::network::{Gradients, Network};

pub trait Optimizer {
    fn step(&mut self, network: &mut Network, gradients: &Gradients);
}

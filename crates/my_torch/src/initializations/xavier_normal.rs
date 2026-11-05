use super::initialization::{Initialization, normal_weights};
use crate::matrix::Matrix;
use crate::rng::Rng;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XavierNormal;

impl Initialization for XavierNormal {
    fn weights(&self, inputs: usize, outputs: usize, rng: &mut Rng) -> Matrix {
        normal_weights(
            inputs,
            outputs,
            |inputs, outputs| (2.0 / (inputs + outputs)).sqrt(),
            rng,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initializations::initialization::assert_spread;

    #[test]
    fn spreads_by_the_square_root_of_two_over_inputs_plus_outputs() {
        let weights = XavierNormal.weights(150, 50, &mut Rng::new(2));
        assert_eq!((weights.rows(), weights.cols()), (150, 50));
        assert_spread(&weights, 0.1);
    }
}

use rand::rngs::StdRng;

use super::initialization::{Initialization, normal_weights};
use crate::matrix::Matrix;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeNormal;

impl Initialization for HeNormal {
    fn weights(&self, inputs: usize, outputs: usize, rng: &mut StdRng) -> Matrix {
        normal_weights(inputs, outputs, |inputs, _| (2.0 / inputs).sqrt(), rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initializations::initialization::assert_spread;
    use rand::SeedableRng;

    #[test]
    fn spreads_by_the_square_root_of_two_over_inputs() {
        let weights = HeNormal.weights(200, 50, &mut StdRng::seed_from_u64(3));
        assert_eq!((weights.rows(), weights.cols()), (200, 50));
        assert_spread(&weights, 0.1);
    }
}

use super::activation::Function;
use crate::matrix::Scalar;

pub(super) struct Identity;

impl Function for Identity {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn value(&self, x: Scalar) -> Scalar {
        x
    }

    fn slope(&self, _x: Scalar) -> Scalar {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::activation::assert_slope_matches_finite_differences;

    #[test]
    fn returns_its_input() {
        for x in [-3.0, 0.0, 2.5] {
            assert_eq!(Identity.value(x), x);
        }
    }

    #[test]
    fn slope_matches_finite_differences() {
        assert_slope_matches_finite_differences(&Identity);
    }
}

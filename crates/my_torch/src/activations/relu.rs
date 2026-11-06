use super::activation::Function;
use crate::matrix::Scalar;

pub(super) struct Relu;

impl Function for Relu {
    fn value(&self, x: Scalar) -> Scalar {
        x.max(0.0)
    }

    fn slope(&self, x: Scalar) -> Scalar {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::activation::assert_slope_matches_finite_differences;

    #[test]
    fn keeps_positive_values_and_zeroes_the_others() {
        assert_eq!(Relu.value(-2.0), 0.0);
        assert_eq!(Relu.value(0.0), 0.0);
        assert_eq!(Relu.value(3.0), 3.0);
    }

    #[test]
    fn slope_is_zero_up_to_zero_and_one_after() {
        assert_eq!(Relu.slope(-1.0), 0.0);
        assert_eq!(Relu.slope(0.0), 0.0);
        assert_eq!(Relu.slope(2.0), 1.0);
    }

    #[test]
    fn slope_matches_finite_differences() {
        assert_slope_matches_finite_differences(&Relu);
    }
}

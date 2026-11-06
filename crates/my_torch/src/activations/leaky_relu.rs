use super::activation::Function;
use crate::matrix::Scalar;

const NEGATIVE_SLOPE: Scalar = 0.01;

pub(super) struct LeakyRelu;

impl Function for LeakyRelu {
    fn name(&self) -> &'static str {
        "leaky_relu"
    }

    fn value(&self, x: Scalar) -> Scalar {
        if x > 0.0 { x } else { NEGATIVE_SLOPE * x }
    }

    fn slope(&self, x: Scalar) -> Scalar {
        if x > 0.0 { 1.0 } else { NEGATIVE_SLOPE }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::activation::assert_slope_matches_finite_differences;

    #[test]
    fn keeps_positive_values_and_shrinks_the_others() {
        assert_eq!(LeakyRelu.value(-2.0), -0.02);
        assert_eq!(LeakyRelu.value(0.0), 0.0);
        assert_eq!(LeakyRelu.value(3.0), 3.0);
    }

    #[test]
    fn slope_is_small_up_to_zero_and_one_after() {
        assert_eq!(LeakyRelu.slope(-1.0), 0.01);
        assert_eq!(LeakyRelu.slope(0.0), 0.01);
        assert_eq!(LeakyRelu.slope(2.0), 1.0);
    }

    #[test]
    fn slope_matches_finite_differences() {
        assert_slope_matches_finite_differences(&LeakyRelu);
    }
}

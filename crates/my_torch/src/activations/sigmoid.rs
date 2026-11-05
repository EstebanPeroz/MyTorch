use super::activation::Function;
use crate::matrix::Scalar;

pub(super) struct Sigmoid;

impl Function for Sigmoid {
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    fn value(&self, x: Scalar) -> Scalar {
        1.0 / (1.0 + (-x).exp())
    }

    fn slope(&self, x: Scalar) -> Scalar {
        let s = self.value(x);
        s * (1.0 - s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::activation::assert_slope_matches_finite_differences;

    #[test]
    fn squashes_values_between_zero_and_one() {
        assert_eq!(Sigmoid.value(0.0), 0.5);
        assert!((Sigmoid.value(2.0) - 0.880_797_1).abs() < 1e-6);
        assert!((Sigmoid.value(-2.0) - 0.119_202_92).abs() < 1e-6);
    }

    #[test]
    fn stays_finite_on_huge_inputs() {
        assert_eq!(Sigmoid.value(-1000.0), 0.0);
        assert_eq!(Sigmoid.value(1000.0), 1.0);
        assert_eq!(Sigmoid.slope(-1000.0), 0.0);
        assert_eq!(Sigmoid.slope(1000.0), 0.0);
    }

    #[test]
    fn slope_matches_finite_differences() {
        assert_slope_matches_finite_differences(&Sigmoid);
    }
}

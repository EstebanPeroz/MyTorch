use super::activation::Function;
use crate::matrix::Scalar;

pub(super) struct Tanh;

impl Function for Tanh {
    fn name(&self) -> &'static str {
        "tanh"
    }

    fn value(&self, x: Scalar) -> Scalar {
        x.tanh()
    }

    fn slope(&self, x: Scalar) -> Scalar {
        1.0 - x.tanh().powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::activation::assert_slope_matches_finite_differences;

    #[test]
    fn squashes_values_between_minus_one_and_one() {
        assert_eq!(Tanh.value(0.0), 0.0);
        assert!((Tanh.value(2.0) - 0.964_027_6).abs() < 1e-6);
        assert!((Tanh.value(-2.0) + 0.964_027_6).abs() < 1e-6);
    }

    #[test]
    fn slope_matches_finite_differences() {
        assert_slope_matches_finite_differences(&Tanh);
    }
}

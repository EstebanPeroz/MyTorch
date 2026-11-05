use super::loss::{Loss, check_shapes};
use crate::matrix::{Matrix, Scalar};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn name(&self) -> &'static str {
        "mean_squared_error"
    }

    fn value(&self, outputs: &Matrix, targets: &Matrix) -> Scalar {
        check_shapes(outputs, targets);
        let squares = outputs.zip_map(targets, |o, t| (o - t) * (o - t));
        squares.data().iter().sum::<Scalar>() / MeanSquaredError::count(outputs)
    }

    fn gradient(&self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        check_shapes(outputs, targets);
        let scale = 2.0 / MeanSquaredError::count(outputs);
        outputs.zip_map(targets, |o, t| scale * (o - t))
    }
}

impl MeanSquaredError {
    fn count(outputs: &Matrix) -> Scalar {
        outputs.data().len() as Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::loss::assert_gradient_matches_finite_differences;

    fn matrix(rows: &[&[Scalar]]) -> Matrix {
        Matrix::from_rows(rows).unwrap()
    }

    #[test]
    fn averages_every_value() {
        let outputs = matrix(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let targets = matrix(&[&[1.0, 0.0], &[3.0, 2.0]]);
        assert_eq!(MeanSquaredError.value(&outputs, &targets), 2.0);
        assert_eq!(
            MeanSquaredError.gradient(&outputs, &targets),
            matrix(&[&[0.0, 1.0], &[0.0, 1.0]])
        );
    }

    #[test]
    fn works_on_any_output_layer() {
        assert!(!MeanSquaredError.needs_raw_scores());
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let outputs = matrix(&[&[0.3, -1.2, 2.0], &[0.5, 0.1, -0.4]]);
        let targets = matrix(&[&[0.0, 0.0, 1.0], &[1.0, 0.5, 0.0]]);
        assert_gradient_matches_finite_differences(&MeanSquaredError, &outputs, &targets);
    }

    #[test]
    #[should_panic(
        expected = "the loss needs targets shaped like the outputs, got a 1 × 2 matrix and a 1 × 3 matrix"
    )]
    fn value_panics_when_outputs_and_targets_differ_in_shape() {
        MeanSquaredError.value(&Matrix::zeros(1, 2), &Matrix::zeros(1, 3));
    }

    #[test]
    #[should_panic(
        expected = "the loss needs targets shaped like the outputs, got a 2 × 2 matrix and a 3 × 2 matrix"
    )]
    fn gradient_panics_when_outputs_and_targets_differ_in_shape() {
        MeanSquaredError.gradient(&Matrix::zeros(2, 2), &Matrix::zeros(3, 2));
    }
}

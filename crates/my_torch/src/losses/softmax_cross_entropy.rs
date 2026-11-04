use super::loss::{Loss, check_shapes};
use crate::matrix::{Matrix, Scalar};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SoftmaxCrossEntropy;

impl Loss for SoftmaxCrossEntropy {
    fn value(&self, outputs: &Matrix, targets: &Matrix) -> Scalar {
        check_shapes(outputs, targets);
        let terms = SoftmaxCrossEntropy::log_softmax(outputs).zip_map(targets, |l, t| -t * l);
        terms.data().iter().sum::<Scalar>() / outputs.rows() as Scalar
    }

    fn gradient(&self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        check_shapes(outputs, targets);
        let scale = 1.0 / outputs.rows() as Scalar;
        let probabilities = SoftmaxCrossEntropy::log_softmax(outputs).map(Scalar::exp);
        let mut values = Vec::with_capacity(outputs.data().len());
        for row in 0..outputs.rows() {
            let total: Scalar = targets.row(row).iter().sum();
            values.extend(
                probabilities
                    .row(row)
                    .iter()
                    .zip(targets.row(row))
                    .map(|(p, t)| scale * (p * total - t)),
            );
        }
        Matrix::from_vec(outputs.rows(), outputs.cols(), values).expect("same shape as the outputs")
    }

    fn needs_raw_scores(&self) -> bool {
        true
    }
}

impl SoftmaxCrossEntropy {
    fn log_softmax(outputs: &Matrix) -> Matrix {
        let mut values = Vec::with_capacity(outputs.data().len());
        for row in 0..outputs.rows() {
            let scores = outputs.row(row);
            let max = scores
                .iter()
                .copied()
                .fold(Scalar::NEG_INFINITY, Scalar::max);
            let log_sum = scores.iter().map(|&x| (x - max).exp()).sum::<Scalar>().ln();
            values.extend(scores.iter().map(|&x| x - max - log_sum));
        }
        Matrix::from_vec(outputs.rows(), outputs.cols(), values).expect("same shape as the outputs")
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
    fn averages_over_samples() {
        let outputs = matrix(&[&[0.0, 0.0], &[0.0, 0.0]]);
        let targets = matrix(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let value = SoftmaxCrossEntropy.value(&outputs, &targets);
        assert!((value - std::f32::consts::LN_2).abs() < 1e-6, "{value}");
        let gradient = SoftmaxCrossEntropy.gradient(&outputs, &targets);
        let expected = [-0.25, 0.25, 0.25, -0.25];
        for (actual, expected) in gradient.data().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{gradient:?}");
        }
    }

    #[test]
    fn is_small_when_the_right_class_wins() {
        let targets = matrix(&[&[0.0, 1.0, 0.0]]);
        let right = SoftmaxCrossEntropy.value(&matrix(&[&[0.0, 5.0, 0.0]]), &targets);
        let wrong = SoftmaxCrossEntropy.value(&matrix(&[&[5.0, 0.0, 0.0]]), &targets);
        assert!(right < 0.02, "{right}");
        assert!(wrong > 5.0, "{wrong}");
    }

    #[test]
    fn needs_raw_scores_since_it_applies_softmax_itself() {
        assert!(SoftmaxCrossEntropy.needs_raw_scores());
    }

    #[test]
    fn stays_finite_on_huge_outputs() {
        let outputs = matrix(&[&[1000.0, -1000.0]]);
        let targets = matrix(&[&[0.0, 1.0]]);
        assert_eq!(SoftmaxCrossEntropy.value(&outputs, &targets), 2000.0);
        assert_eq!(
            SoftmaxCrossEntropy.gradient(&outputs, &targets),
            matrix(&[&[1.0, -1.0]])
        );
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let outputs = matrix(&[&[0.3, -1.2, 2.0], &[0.5, 0.1, -0.4], &[-0.7, 1.1, 0.2]]);
        let targets = matrix(&[&[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0], &[0.5, 0.0, 1.0]]);
        assert_gradient_matches_finite_differences(&SoftmaxCrossEntropy, &outputs, &targets);
    }

    #[test]
    #[should_panic(
        expected = "the loss needs targets shaped like the outputs, got a 1 × 2 matrix and a 1 × 3 matrix"
    )]
    fn value_panics_when_outputs_and_targets_differ_in_shape() {
        SoftmaxCrossEntropy.value(&Matrix::zeros(1, 2), &Matrix::zeros(1, 3));
    }

    #[test]
    #[should_panic(
        expected = "the loss needs targets shaped like the outputs, got a 2 × 2 matrix and a 2 × 3 matrix"
    )]
    fn gradient_panics_when_outputs_and_targets_differ_in_shape() {
        SoftmaxCrossEntropy.gradient(&Matrix::zeros(2, 2), &Matrix::zeros(2, 3));
    }
}

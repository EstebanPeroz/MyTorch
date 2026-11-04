use crate::matrix::{Matrix, Scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    MeanSquaredError,
    SoftmaxCrossEntropy,
}

impl Loss {
    pub fn value(self, outputs: &Matrix, targets: &Matrix) -> Scalar {
        Loss::check_shapes(outputs, targets);
        let terms = match self {
            Loss::MeanSquaredError => outputs.zip_map(targets, |o, t| (o - t) * (o - t)),
            Loss::SoftmaxCrossEntropy => log_softmax(outputs).zip_map(targets, |l, t| -t * l),
        };
        terms.data().iter().sum::<Scalar>() / self.divisor(outputs)
    }

    pub fn gradient(self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        Loss::check_shapes(outputs, targets);
        let scale = 1.0 / self.divisor(outputs);
        match self {
            Loss::MeanSquaredError => outputs.zip_map(targets, |o, t| 2.0 * scale * (o - t)),
            Loss::SoftmaxCrossEntropy => {
                let probabilities = log_softmax(outputs).map(Scalar::exp);
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
                Matrix::from_vec(outputs.rows(), outputs.cols(), values)
                    .expect("same shape as the outputs")
            }
        }
    }

    fn divisor(self, outputs: &Matrix) -> Scalar {
        match self {
            Loss::MeanSquaredError => outputs.data().len() as Scalar,
            Loss::SoftmaxCrossEntropy => outputs.rows() as Scalar,
        }
    }

    fn check_shapes(outputs: &Matrix, targets: &Matrix) {
        assert!(
            outputs.rows() == targets.rows() && outputs.cols() == targets.cols(),
            "the loss needs targets shaped like the outputs, got a {outputs} and a {targets}"
        );
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Loss; 2] = [Loss::MeanSquaredError, Loss::SoftmaxCrossEntropy];

    fn matrix(rows: &[&[Scalar]]) -> Matrix {
        Matrix::from_rows(rows).unwrap()
    }

    fn assert_close(actual: &Matrix, expected: &Matrix, tolerance: Scalar) {
        assert_eq!(
            (actual.rows(), actual.cols()),
            (expected.rows(), expected.cols())
        );
        for (a, e) in actual.data().iter().zip(expected.data()) {
            assert!((a - e).abs() < tolerance, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn mean_squared_error_averages_every_value() {
        let outputs = matrix(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let targets = matrix(&[&[1.0, 0.0], &[3.0, 2.0]]);
        assert_eq!(Loss::MeanSquaredError.value(&outputs, &targets), 2.0);
        assert_eq!(
            Loss::MeanSquaredError.gradient(&outputs, &targets),
            matrix(&[&[0.0, 1.0], &[0.0, 1.0]])
        );
    }

    #[test]
    fn cross_entropy_averages_over_samples() {
        let outputs = matrix(&[&[0.0, 0.0], &[0.0, 0.0]]);
        let targets = matrix(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let value = Loss::SoftmaxCrossEntropy.value(&outputs, &targets);
        assert!((value - std::f32::consts::LN_2).abs() < 1e-6, "{value}");
        assert_close(
            &Loss::SoftmaxCrossEntropy.gradient(&outputs, &targets),
            &matrix(&[&[-0.25, 0.25], &[0.25, -0.25]]),
            1e-6,
        );
    }

    #[test]
    fn cross_entropy_is_small_when_the_right_class_wins() {
        let targets = matrix(&[&[0.0, 1.0, 0.0]]);
        let right = Loss::SoftmaxCrossEntropy.value(&matrix(&[&[0.0, 5.0, 0.0]]), &targets);
        let wrong = Loss::SoftmaxCrossEntropy.value(&matrix(&[&[5.0, 0.0, 0.0]]), &targets);
        assert!(right < 0.02, "{right}");
        assert!(wrong > 5.0, "{wrong}");
    }

    #[test]
    fn cross_entropy_stays_finite_on_huge_outputs() {
        let outputs = matrix(&[&[1000.0, -1000.0]]);
        let targets = matrix(&[&[0.0, 1.0]]);
        assert_eq!(Loss::SoftmaxCrossEntropy.value(&outputs, &targets), 2000.0);
        assert_eq!(
            Loss::SoftmaxCrossEntropy.gradient(&outputs, &targets),
            matrix(&[&[1.0, -1.0]])
        );
    }

    #[test]
    fn gradients_match_finite_differences() {
        let outputs = matrix(&[&[0.3, -1.2, 2.0], &[0.5, 0.1, -0.4]]);
        let targets = matrix(&[&[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]]);
        let step = 1e-2;
        for loss in ALL {
            let mut numeric = Matrix::zeros(outputs.rows(), outputs.cols());
            for row in 0..outputs.rows() {
                for col in 0..outputs.cols() {
                    let mut above = outputs.clone();
                    above.set(row, col, outputs.get(row, col) + step);
                    let mut below = outputs.clone();
                    below.set(row, col, outputs.get(row, col) - step);
                    let slope = (loss.value(&above, &targets) - loss.value(&below, &targets))
                        / (2.0 * step);
                    numeric.set(row, col, slope);
                }
            }
            assert_close(&loss.gradient(&outputs, &targets), &numeric, 1e-3);
        }
    }

    #[test]
    #[should_panic(
        expected = "the loss needs targets shaped like the outputs, got a 1 × 2 matrix and a 1 × 3 matrix"
    )]
    fn value_panics_when_outputs_and_targets_differ_in_shape() {
        Loss::MeanSquaredError.value(&Matrix::zeros(1, 2), &Matrix::zeros(1, 3));
    }

    #[test]
    #[should_panic(
        expected = "the loss needs targets shaped like the outputs, got a 2 × 2 matrix and a 2 × 3 matrix"
    )]
    fn gradient_panics_when_outputs_and_targets_differ_in_shape() {
        Loss::SoftmaxCrossEntropy.gradient(&Matrix::zeros(2, 2), &Matrix::zeros(2, 3));
    }
}

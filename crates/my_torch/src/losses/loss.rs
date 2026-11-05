use super::mean_squared_error::MeanSquaredError;
use super::softmax_cross_entropy::SoftmaxCrossEntropy;
use crate::matrix::{Matrix, Scalar};
use crate::names::find_by_name;

pub trait Loss {
    fn name(&self) -> &'static str;

    fn value(&self, outputs: &Matrix, targets: &Matrix) -> Scalar;

    fn gradient(&self, outputs: &Matrix, targets: &Matrix) -> Matrix;

    fn needs_raw_scores(&self) -> bool {
        false
    }
}

impl<L: Loss + ?Sized> Loss for Box<L> {
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    fn value(&self, outputs: &Matrix, targets: &Matrix) -> Scalar {
        self.as_ref().value(outputs, targets)
    }

    fn gradient(&self, outputs: &Matrix, targets: &Matrix) -> Matrix {
        self.as_ref().gradient(outputs, targets)
    }

    fn needs_raw_scores(&self) -> bool {
        self.as_ref().needs_raw_scores()
    }
}

pub fn loss_from_name(name: &str) -> Result<Box<dyn Loss>, String> {
    let all: Vec<Box<dyn Loss>> = vec![Box::new(MeanSquaredError), Box::new(SoftmaxCrossEntropy)];
    find_by_name("loss", name, all, |loss| loss.name())
}

pub(super) fn check_shapes(outputs: &Matrix, targets: &Matrix) {
    assert!(
        outputs.rows() == targets.rows() && outputs.cols() == targets.cols(),
        "the loss needs targets shaped like the outputs, got a {outputs} and a {targets}"
    );
}

#[cfg(test)]
pub(super) fn assert_gradient_matches_finite_differences(
    loss: &impl Loss,
    outputs: &Matrix,
    targets: &Matrix,
) {
    let step = 1e-2;
    let gradient = loss.gradient(outputs, targets);
    for row in 0..outputs.rows() {
        for col in 0..outputs.cols() {
            let mut above = outputs.clone();
            above.set(row, col, outputs.get(row, col) + step);
            let mut below = outputs.clone();
            below.set(row, col, outputs.get(row, col) - step);
            let numeric =
                (loss.value(&above, targets) - loss.value(&below, targets)) / (2.0 * step);
            let analytic = gradient.get(row, col);
            assert!(
                (numeric - analytic).abs() < 1e-3,
                "({row}, {col}): measured {numeric}, gradient gave {analytic}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_a_loss(loss: &impl Loss) -> (&'static str, Scalar, Matrix, bool) {
        let outputs = Matrix::from_rows(&[[0.5, -1.0]]).unwrap();
        let targets = Matrix::from_rows(&[[1.0, 0.0]]).unwrap();
        (
            loss.name(),
            loss.value(&outputs, &targets),
            loss.gradient(&outputs, &targets),
            loss.needs_raw_scores(),
        )
    }

    #[test]
    fn a_loss_found_by_name_behaves_like_the_loss_itself() {
        assert_eq!(
            as_a_loss(&loss_from_name("mean_squared_error").unwrap()),
            as_a_loss(&MeanSquaredError)
        );
        assert_eq!(
            as_a_loss(&loss_from_name("softmax_cross_entropy").unwrap()),
            as_a_loss(&SoftmaxCrossEntropy)
        );
    }

    #[test]
    fn rejects_an_unknown_loss_name() {
        assert_eq!(
            loss_from_name("huber").err(),
            Some(
                "unknown loss \"huber\", expected one of: mean_squared_error, softmax_cross_entropy"
                    .to_string()
            )
        );
    }
}

use crate::matrix::{Matrix, Scalar};

pub trait Loss {
    fn value(&self, outputs: &Matrix, targets: &Matrix) -> Scalar;

    fn gradient(&self, outputs: &Matrix, targets: &Matrix) -> Matrix;
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

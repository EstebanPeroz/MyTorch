use crate::matrix::{Matrix, Scalar};

const LEAKY_RELU_SLOPE: Scalar = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
}

impl Activation {
    pub fn apply(self, z: &Matrix) -> Matrix {
        z.map(|x| self.value(x))
    }

    pub fn derivative(self, z: &Matrix) -> Matrix {
        z.map(|x| self.slope(x))
    }

    fn value(self, x: Scalar) -> Scalar {
        match self {
            Activation::Identity => x,
            Activation::Sigmoid => sigmoid(x),
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
            Activation::LeakyRelu => {
                if x > 0.0 {
                    x
                } else {
                    LEAKY_RELU_SLOPE * x
                }
            }
        }
    }

    fn slope(self, x: Scalar) -> Scalar {
        match self {
            Activation::Identity => 1.0,
            Activation::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyRelu => {
                if x > 0.0 {
                    1.0
                } else {
                    LEAKY_RELU_SLOPE
                }
            }
        }
    }
}

fn sigmoid(x: Scalar) -> Scalar {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Activation; 5] = [
        Activation::Identity,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Relu,
        Activation::LeakyRelu,
    ];

    fn row(values: &[Scalar]) -> Matrix {
        Matrix::from_rows(&[values]).unwrap()
    }

    #[test]
    fn computes_known_values() {
        let z = row(&[-2.0, 0.0, 3.0]);
        assert_eq!(Activation::Identity.apply(&z), z);
        assert_eq!(Activation::Relu.apply(&z), row(&[0.0, 0.0, 3.0]));
        assert_eq!(Activation::LeakyRelu.apply(&z), row(&[-0.02, 0.0, 3.0]));
        assert_eq!(Activation::Sigmoid.apply(&row(&[0.0])), row(&[0.5]));
        assert_eq!(Activation::Tanh.apply(&row(&[0.0])), row(&[0.0]));
    }

    #[test]
    fn sigmoid_stays_finite_on_huge_inputs() {
        let z = row(&[-1000.0, 1000.0]);
        assert_eq!(Activation::Sigmoid.apply(&z), row(&[0.0, 1.0]));
        assert_eq!(Activation::Sigmoid.derivative(&z), row(&[0.0, 0.0]));
    }

    #[test]
    fn relu_slopes_are_zero_or_one_away_from_zero() {
        let z = row(&[-1.0, 0.0, 2.0]);
        assert_eq!(Activation::Relu.derivative(&z), row(&[0.0, 0.0, 1.0]));
        assert_eq!(
            Activation::LeakyRelu.derivative(&z),
            row(&[0.01, 0.01, 1.0])
        );
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let step = 1e-2;
        let z = row(&[-2.5, -0.7, -0.1, 0.3, 1.2, 3.0]);
        for activation in ALL {
            let above = activation.apply(&z.map(|x| x + step));
            let below = activation.apply(&z.map(|x| x - step));
            let numeric = above.zip_map(&below, |a, b| (a - b) / (2.0 * step));
            let analytic = activation.derivative(&z);
            for (n, a) in numeric.data().iter().zip(analytic.data()) {
                assert!((n - a).abs() < 1e-3, "{activation:?}: {n} vs {a}");
            }
        }
    }
}

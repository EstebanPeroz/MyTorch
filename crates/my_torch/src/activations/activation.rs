use super::identity::Identity;
use super::leaky_relu::LeakyRelu;
use super::relu::Relu;
use super::sigmoid::Sigmoid;
use super::tanh::Tanh;
use crate::matrix::{Matrix, Scalar};
use crate::names::find_by_name;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
}

pub(super) trait Function {
    fn name(&self) -> &'static str;

    fn value(&self, x: Scalar) -> Scalar;

    fn slope(&self, x: Scalar) -> Scalar;
}

impl Activation {
    pub const ALL: [Activation; 5] = [
        Activation::Identity,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Relu,
        Activation::LeakyRelu,
    ];

    pub fn name(self) -> &'static str {
        self.function().name()
    }

    pub fn from_name(name: &str) -> Result<Activation, String> {
        find_by_name("activation", name, Activation::ALL.to_vec(), |activation| {
            activation.name()
        })
    }

    pub fn apply(self, z: &Matrix) -> Matrix {
        let function = self.function();
        z.map(|x| function.value(x))
    }

    pub fn derivative(self, z: &Matrix) -> Matrix {
        let function = self.function();
        z.map(|x| function.slope(x))
    }

    fn function(self) -> &'static dyn Function {
        match self {
            Activation::Identity => &Identity,
            Activation::Sigmoid => &Sigmoid,
            Activation::Tanh => &Tanh,
            Activation::Relu => &Relu,
            Activation::LeakyRelu => &LeakyRelu,
        }
    }
}

#[cfg(test)]
pub(super) fn assert_slope_matches_finite_differences(function: &impl Function) {
    let step = 1e-2;
    for x in [-2.5, -0.7, -0.1, 0.3, 1.2, 3.0] {
        let numeric = (function.value(x + step) - function.value(x - step)) / (2.0 * step);
        let analytic = function.slope(x);
        assert!(
            (numeric - analytic).abs() < 1e-3,
            "at {x}: measured {numeric}, slope gave {analytic}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_activation_is_found_by_its_name() {
        let names: Vec<&str> = Activation::ALL.iter().map(|a| a.name()).collect();
        assert_eq!(names, ["identity", "sigmoid", "tanh", "relu", "leaky_relu"]);
        for activation in Activation::ALL {
            assert_eq!(Activation::from_name(activation.name()), Ok(activation));
        }
    }

    #[test]
    fn rejects_an_unknown_activation_name() {
        assert_eq!(
            Activation::from_name("softmax"),
            Err("unknown activation \"softmax\", expected one of: identity, sigmoid, tanh, relu, leaky_relu"
                .to_string())
        );
    }

    #[test]
    fn each_activation_uses_its_own_function_on_every_value() {
        let z = Matrix::from_rows(&[[-2.0, -2.0], [-2.0, -2.0]]).unwrap();
        let cases = [
            (Activation::Identity, -2.0, 1.0),
            (Activation::Sigmoid, 0.119_202_92, 0.104_993_59),
            (Activation::Tanh, -0.964_027_6, 0.070_650_82),
            (Activation::Relu, 0.0, 0.0),
            (Activation::LeakyRelu, -0.02, 0.01),
        ];
        for (activation, value, slope) in cases {
            for (actual, expected) in [
                (activation.apply(&z), value),
                (activation.derivative(&z), slope),
            ] {
                assert_eq!((actual.rows(), actual.cols()), (2, 2));
                assert!(
                    actual.data().iter().all(|a| (a - expected).abs() < 1e-6),
                    "{activation:?}: {actual:?}, expected {expected}"
                );
            }
        }
    }
}

mod activations;
mod dense;
mod initializations;
mod losses;
mod matrix;

pub use activations::Activation;
pub use dense::{Dense, DenseCache, DenseGradients};
pub use initializations::{HeNormal, Initialization, XavierNormal};
pub use losses::{Loss, MeanSquaredError, SoftmaxCrossEntropy};
pub use matrix::{Matrix, Scalar};

mod activations;
mod dense;
mod initializations;
mod loss;
mod matrix;

pub use activations::Activation;
pub use dense::{Dense, DenseCache, DenseGradients};
pub use initializations::{HeNormal, Initialization, XavierNormal};
pub use loss::Loss;
pub use matrix::{Matrix, Scalar};

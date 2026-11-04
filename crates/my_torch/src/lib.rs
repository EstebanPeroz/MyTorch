mod activations;
mod dense;
mod initializations;
mod matrix;

pub use activations::Activation;
pub use dense::{Dense, DenseCache, DenseGradients};
pub use initializations::{HeNormal, Initialization, XavierNormal};
pub use matrix::{Matrix, Scalar};

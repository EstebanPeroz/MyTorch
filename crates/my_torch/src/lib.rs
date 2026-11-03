mod activation;
mod dense;
mod initialization;
mod matrix;

pub use activation::Activation;
pub use dense::{Dense, DenseCache, DenseGradients};
pub use initialization::Initialization;
pub use matrix::{Matrix, Scalar};

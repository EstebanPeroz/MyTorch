use crate::matrix::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    inputs: Matrix,
    targets: Matrix,
}

impl Dataset {
    pub fn new(inputs: Matrix, targets: Matrix) -> Result<Dataset, String> {
        if inputs.rows() != targets.rows() {
            return Err(format!(
                "{} samples have inputs but {} have targets",
                inputs.rows(),
                targets.rows()
            ));
        }
        Ok(Dataset { inputs, targets })
    }

    pub fn inputs(&self) -> &Matrix {
        &self.inputs
    }

    pub fn targets(&self) -> &Matrix {
        &self.targets
    }

    pub fn sample_count(&self) -> usize {
        self.inputs.rows()
    }

    pub fn select(&self, indices: &[usize]) -> Dataset {
        Dataset {
            inputs: self.inputs.select_rows(indices),
            targets: self.targets.select_rows(indices),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> Dataset {
        Dataset::new(
            Matrix::from_rows(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]).unwrap(),
            Matrix::from_rows(&[[0.0], [1.0], [1.0]]).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn needs_one_target_row_per_input_row() {
        assert_eq!(
            Dataset::new(Matrix::zeros(3, 2), Matrix::zeros(4, 1)),
            Err("3 samples have inputs but 4 have targets".to_string())
        );
    }

    #[test]
    fn counts_samples() {
        assert_eq!(dataset().sample_count(), 3);
    }

    #[test]
    fn select_keeps_inputs_and_targets_together() {
        let selected = dataset().select(&[2, 0]);
        assert_eq!(
            selected.inputs(),
            &Matrix::from_rows(&[[1.0, 0.0], [0.0, 0.0]]).unwrap()
        );
        assert_eq!(
            selected.targets(),
            &Matrix::from_rows(&[[1.0], [0.0]]).unwrap()
        );
    }
}

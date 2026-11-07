use crate::matrix::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    inputs: Matrix,
    targets: Matrix,
    groups: Option<Vec<usize>>,
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
        Ok(Dataset {
            inputs,
            targets,
            groups: None,
        })
    }

    /// Tells which samples belong together, with one group number per sample, in the order of the rows. A
    /// validation split never separates a group: all its samples are used for training or all for validation.
    /// This is for samples that are copies of each other (a mirrored chessboard, for example), which would
    /// otherwise leak from the training set into the validation set.
    pub fn with_groups(mut self, groups: Vec<usize>) -> Result<Dataset, String> {
        if groups.len() != self.sample_count() {
            return Err(format!(
                "{} samples were given {} groups",
                self.sample_count(),
                groups.len()
            ));
        }
        self.groups = Some(groups);
        Ok(self)
    }

    pub fn groups(&self) -> Option<&[usize]> {
        self.groups.as_deref()
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
            groups: None,
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
    fn needs_one_group_per_sample() {
        assert_eq!(dataset().groups(), None);
        assert_eq!(
            dataset().with_groups(vec![0, 1]),
            Err("3 samples were given 2 groups".to_string())
        );
        let grouped = dataset().with_groups(vec![0, 1, 0]).unwrap();
        assert_eq!(grouped.groups(), Some(&[0, 1, 0][..]));
        assert_eq!(grouped.select(&[0, 1]).groups(), None);
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

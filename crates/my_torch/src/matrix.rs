use std::fmt;

use ndarray::{Array2, ArrayView2, Axis, Zip, linalg::general_mat_mul};
use rayon::prelude::*;

pub type Scalar = f32;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    values: Array2<Scalar>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        assert!(rows > 0 && cols > 0, "a matrix cannot be {rows} × {cols}");
        Matrix::wrap(Array2::zeros((rows, cols)))
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<Scalar>) -> Result<Matrix, String> {
        if rows == 0 || cols == 0 {
            return Err(format!("a matrix cannot be {rows} × {cols}"));
        }
        if data.len() != rows * cols {
            return Err(format!(
                "a {rows} × {cols} matrix needs {} values, got {}",
                rows * cols,
                data.len()
            ));
        }
        let values = Array2::from_shape_vec((rows, cols), data).expect("the shape was checked");
        Ok(Matrix::wrap(values))
    }

    pub fn from_rows<R: AsRef<[Scalar]>>(rows: &[R]) -> Result<Matrix, String> {
        let cols = rows.first().map_or(0, |row| row.as_ref().len());
        if let Some(index) = rows.iter().position(|row| row.as_ref().len() != cols) {
            return Err(format!(
                "matrix row {} has {} values, expected {cols}",
                index + 1,
                rows[index].as_ref().len()
            ));
        }
        let data = rows.iter().flat_map(AsRef::as_ref).copied().collect();
        Matrix::from_vec(rows.len(), cols, data)
    }

    pub fn rows(&self) -> usize {
        self.values.nrows()
    }

    pub fn cols(&self) -> usize {
        self.values.ncols()
    }

    pub fn data(&self) -> &[Scalar] {
        self.values
            .as_slice()
            .expect("a matrix is always stored row after row")
    }

    pub fn row(&self, row: usize) -> &[Scalar] {
        assert!(row < self.rows(), "row {row} is outside {self}");
        &self.data()[row * self.cols()..(row + 1) * self.cols()]
    }

    pub fn get(&self, row: usize, col: usize) -> Scalar {
        self.check_index(row, col);
        self.values[[row, col]]
    }

    pub fn set(&mut self, row: usize, col: usize, value: Scalar) {
        self.check_index(row, col);
        self.values[[row, col]] = value;
    }

    pub fn argmax_rows(&self) -> Vec<usize> {
        self.values
            .rows()
            .into_iter()
            .map(|values| {
                (1..values.len()).fold(0, |best, col| {
                    if values[col] > values[best] {
                        col
                    } else {
                        best
                    }
                })
            })
            .collect()
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols(), other.rows(), "matmul: {self} · {other}");
        if is_sparse(&self.values) {
            return Matrix::wrap(sparse_dot(&self.values, &other.values));
        }
        Matrix::wrap(parallel_dot(self.values.view(), other.values.view()))
    }

    pub fn transpose_matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.rows(),
            other.rows(),
            "transpose_matmul: {self}ᵀ · {other}"
        );
        if is_sparse(&self.values) {
            return Matrix::wrap(sparse_transpose_dot(&self.values, &other.values));
        }
        Matrix::wrap(parallel_dot(self.values.t(), other.values.view()))
    }

    pub fn matmul_transpose(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols(),
            other.cols(),
            "matmul_transpose: {self} · {other}ᵀ"
        );
        Matrix::wrap(parallel_dot(self.values.view(), other.values.t()))
    }

    pub fn add_row(&self, row: &Matrix) -> Matrix {
        assert!(
            row.rows() == 1 && row.cols() == self.cols(),
            "add_row: {self} + {row}"
        );
        Matrix::wrap(&self.values + &row.values)
    }

    pub fn sum_rows(&self) -> Matrix {
        Matrix::wrap(self.values.sum_axis(Axis(0)).insert_axis(Axis(0)))
    }

    pub fn map(&self, function: impl Fn(Scalar) -> Scalar) -> Matrix {
        Matrix::wrap(self.values.mapv(function))
    }

    pub fn zip_map(&self, other: &Matrix, function: impl Fn(Scalar, Scalar) -> Scalar) -> Matrix {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "zip_map: {self} and {other}"
        );
        Matrix::wrap(
            Zip::from(&self.values)
                .and(&other.values)
                .map_collect(|&left, &right| function(left, right)),
        )
    }

    pub fn zip_apply(&mut self, other: &Matrix, function: impl Fn(Scalar, Scalar) -> Scalar) {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "zip_apply: {self} and {other}"
        );
        Zip::from(&mut self.values)
            .and(&other.values)
            .for_each(|value, &other| *value = function(*value, other));
    }

    pub fn apply(&mut self, function: impl Fn(Scalar) -> Scalar) {
        self.values.mapv_inplace(function);
    }

    pub fn slice_rows(&self, start: usize, end: usize) -> Matrix {
        assert!(
            start < end && end <= self.rows(),
            "slice_rows: rows {start}..{end} are outside {self}"
        );
        Matrix::wrap(self.values.slice(ndarray::s![start..end, ..]).to_owned())
    }

    pub fn select_rows(&self, indices: &[usize]) -> Matrix {
        assert!(!indices.is_empty(), "select_rows: no rows selected");
        if let Some(row) = indices.iter().find(|&&row| row >= self.rows()) {
            panic!("select_rows: row {row} is outside {self}");
        }
        Matrix::wrap(self.values.select(Axis(0), indices))
    }

    fn wrap(values: Array2<Scalar>) -> Matrix {
        let values = if values.is_standard_layout() {
            values
        } else {
            values.as_standard_layout().into_owned()
        };
        Matrix { values }
    }

    fn check_index(&self, row: usize, col: usize) {
        assert!(
            row < self.rows() && col < self.cols(),
            "({row}, {col}) is outside {self}"
        );
    }
}

/// A matrix is sparse when at most one value in `SPARSE_RATIO` is not zero: a chessboard has at most 32 pieces
/// among its 769 inputs, so the first layer reads very sparse inputs, unlike the next ones.
const SPARSE_RATIO: usize = 8;
/// Rows of the output computed by one task of a sparse product.
const SPARSE_BLOCK_ROWS: usize = 64;

fn is_sparse(values: &Array2<Scalar>) -> bool {
    let non_zeros = values.iter().filter(|&&value| value != 0.0).count();
    non_zeros * SPARSE_RATIO <= values.len()
}

/// `left · right`, adding only the rows of `right` that a non-zero value of `left` selects.
fn sparse_dot(left: &Array2<Scalar>, right: &Array2<Scalar>) -> Array2<Scalar> {
    let mut product = Array2::zeros((left.nrows(), right.ncols()));
    let mut rows: Vec<_> = product.axis_iter_mut(Axis(0)).collect();
    rows.par_iter_mut().enumerate().for_each(|(index, row)| {
        for (inner, &value) in left.row(index).iter().enumerate() {
            if value != 0.0 {
                row.scaled_add(value, &right.row(inner));
            }
        }
    });
    product
}

/// `leftᵀ · right`, adding only the rows of `right` that a non-zero value of `left` selects.
fn sparse_transpose_dot(left: &Array2<Scalar>, right: &Array2<Scalar>) -> Array2<Scalar> {
    let mut product = Array2::zeros((left.ncols(), right.ncols()));
    let mut blocks: Vec<_> = product
        .axis_chunks_iter_mut(Axis(0), SPARSE_BLOCK_ROWS)
        .collect();
    blocks
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, block)| {
            let start = index * SPARSE_BLOCK_ROWS;
            for sample in 0..left.nrows() {
                for row in 0..block.nrows() {
                    let value = left[[sample, start + row]];
                    if value != 0.0 {
                        block.row_mut(row).scaled_add(value, &right.row(sample));
                    }
                }
            }
        });
    product
}

/// Output columns computed by one task when a product is split across threads.
const PRODUCT_COLUMNS: usize = 32;
/// Products smaller than this many multiply-adds are not worth the threads.
const PARALLEL_PRODUCT_WORK: usize = 1 << 20;

/// `left · right`, with the output columns split across the rayon threads. Every output value is still one
/// plain dot product, so the result is the same as a single `dot`, whatever the number of threads.
fn parallel_dot(left: ArrayView2<Scalar>, right: ArrayView2<Scalar>) -> Array2<Scalar> {
    let (rows, inner, cols) = (left.nrows(), left.ncols(), right.ncols());
    if rows * inner * cols < PARALLEL_PRODUCT_WORK || cols < 2 * PRODUCT_COLUMNS {
        return left.dot(&right);
    }
    let mut product = Array2::zeros((rows, cols));
    let mut blocks: Vec<_> = product
        .axis_chunks_iter_mut(Axis(1), PRODUCT_COLUMNS)
        .collect();
    blocks
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, block)| {
            let start = index * PRODUCT_COLUMNS;
            let right = right.slice(ndarray::s![.., start..start + block.ncols()]);
            general_mat_mul(1.0, &left, &right, 0.0, block);
        });
    product
}

impl fmt::Display for Matrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} × {} matrix", self.rows(), self.cols())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_products_give_the_same_values_as_dense_ones() {
        // Three values per row of 300 are not zero, so the left operand takes the sparse path.
        let sparse = |rows: usize| {
            let mut m = Matrix::zeros(rows, 300);
            for row in 0..rows {
                for step in 0..3 {
                    m.set(
                        row,
                        (row * 31 + step * 97) % 300,
                        (row + step) as Scalar - 2.5,
                    );
                }
            }
            m
        };
        let dense = |rows: usize, cols: usize| {
            let data = (0..rows * cols).map(|i| (i % 11) as Scalar * 0.25 - 1.0);
            Matrix::from_vec(rows, cols, data.collect()).unwrap()
        };
        let (left, right) = (sparse(10), dense(300, 70));
        assert!(is_sparse(&left.values));
        assert_eq!(
            left.matmul(&right),
            Matrix::wrap(left.values.dot(&right.values))
        );
        let right = dense(10, 70);
        assert_eq!(
            left.transpose_matmul(&right),
            Matrix::wrap(left.values.t().dot(&right.values))
        );
    }

    #[test]
    fn large_products_split_across_threads_give_the_same_values() {
        let value = |row: usize, col: usize| ((row * 7 + col * 13) % 17) as Scalar - 8.0;
        let build = |rows: usize, cols: usize| {
            let data = (0..rows * cols)
                .map(|i| value(i / cols, i % cols))
                .collect();
            Matrix::from_vec(rows, cols, data).unwrap()
        };
        let (a, b) = (build(32, 300), build(300, 200));
        let expected = Matrix::wrap(a.values.dot(&b.values));
        assert_eq!(a.matmul(&b), expected);
        assert_eq!(
            build(300, 32).transpose_matmul(&b),
            Matrix::wrap(build(300, 32).values.t().dot(&b.values))
        );
        assert_eq!(
            a.matmul_transpose(&build(200, 300)),
            Matrix::wrap(a.values.dot(&build(200, 300).values.t()))
        );
    }

    fn matrix(rows: &[&[Scalar]]) -> Matrix {
        Matrix::from_rows(rows).unwrap()
    }

    #[test]
    fn from_rows_reads_row_by_row() {
        let m = matrix(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        assert_eq!((m.rows(), m.cols()), (2, 3));
        assert_eq!(m.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(m.get(0, 2), 3.0);
    }

    #[test]
    fn from_rows_rejects_rows_of_different_lengths() {
        assert_eq!(
            Matrix::from_rows(&[&[1.0, 2.0][..], &[3.0]]),
            Err("matrix row 2 has 1 values, expected 2".to_string())
        );
    }

    #[test]
    fn rejects_empty_matrices() {
        let no_rows: &[&[Scalar]] = &[];
        assert_eq!(
            Matrix::from_rows(no_rows),
            Err("a matrix cannot be 0 × 0".to_string())
        );
        assert_eq!(
            Matrix::from_rows(&[&[] as &[Scalar]]),
            Err("a matrix cannot be 1 × 0".to_string())
        );
        assert_eq!(
            Matrix::from_vec(0, 3, vec![]),
            Err("a matrix cannot be 0 × 3".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "a matrix cannot be 2 × 0")]
    fn zeros_panics_on_an_empty_shape() {
        Matrix::zeros(2, 0);
    }

    #[test]
    fn from_vec_rejects_the_wrong_number_of_values() {
        assert_eq!(
            Matrix::from_vec(2, 3, vec![0.0; 5]),
            Err("a 2 × 3 matrix needs 6 values, got 5".to_string())
        );
    }

    #[test]
    fn set_changes_one_value() {
        let mut m = Matrix::zeros(2, 2);
        m.set(1, 0, 7.0);
        assert_eq!(m, matrix(&[&[0.0, 0.0], &[7.0, 0.0]]));
    }

    #[test]
    #[should_panic(expected = "(2, 0) is outside 2 × 2 matrix")]
    fn get_panics_outside_the_matrix() {
        Matrix::zeros(2, 2).get(2, 0);
    }

    #[test]
    #[should_panic(expected = "row 2 is outside 2 × 3 matrix")]
    fn row_panics_outside_the_matrix() {
        Matrix::zeros(2, 3).row(2);
    }

    #[test]
    fn matmul_multiplies_rows_by_columns() {
        let a = matrix(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let b = matrix(&[&[1.0, 0.0, -1.0], &[2.0, 1.0, 0.0]]);
        assert_eq!(
            a.matmul(&b),
            matrix(&[&[5.0, 2.0, -1.0], &[11.0, 4.0, -3.0], &[17.0, 6.0, -5.0]])
        );
    }

    #[test]
    fn transpose_matmul_transposes_the_left_matrix() {
        let a = matrix(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let b = matrix(&[&[1.0], &[0.0], &[-1.0]]);
        assert_eq!(a.transpose_matmul(&b), matrix(&[&[-4.0], &[-4.0]]));
    }

    #[test]
    fn matmul_transpose_transposes_the_right_matrix() {
        let a = matrix(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = matrix(&[&[1.0, 1.0], &[0.0, -1.0], &[2.0, 0.0]]);
        assert_eq!(
            a.matmul_transpose(&b),
            matrix(&[&[3.0, -2.0, 2.0], &[7.0, -4.0, 6.0]])
        );
    }

    #[test]
    #[should_panic(expected = "matmul: 2 × 3 matrix · 2 × 3 matrix")]
    fn matmul_panics_on_mismatched_shapes() {
        Matrix::zeros(2, 3).matmul(&Matrix::zeros(2, 3));
    }

    #[test]
    fn add_row_adds_to_every_row() {
        let a = matrix(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let bias = matrix(&[&[10.0, -1.0]]);
        assert_eq!(a.add_row(&bias), matrix(&[&[11.0, 1.0], &[13.0, 3.0]]));
    }

    #[test]
    #[should_panic(expected = "add_row: 2 × 2 matrix + 2 × 1 matrix")]
    fn add_row_panics_on_a_column() {
        Matrix::zeros(2, 2).add_row(&Matrix::zeros(2, 1));
    }

    #[test]
    fn sum_rows_adds_rows_together() {
        let a = matrix(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        assert_eq!(a.sum_rows(), matrix(&[&[9.0, 12.0]]));
    }

    #[test]
    fn map_and_zip_map_work_value_by_value() {
        let a = matrix(&[&[1.0, -2.0]]);
        let b = matrix(&[&[3.0, 4.0]]);
        assert_eq!(a.map(|x| x * 2.0), matrix(&[&[2.0, -4.0]]));
        assert_eq!(a.zip_map(&b, |x, y| x * y), matrix(&[&[3.0, -8.0]]));
    }

    #[test]
    #[should_panic(expected = "zip_map: 1 × 2 matrix and 2 × 1 matrix")]
    fn zip_map_panics_on_mismatched_shapes() {
        Matrix::zeros(1, 2).zip_map(&Matrix::zeros(2, 1), |x, _| x);
    }

    #[test]
    fn zip_apply_updates_in_place() {
        let mut a = matrix(&[&[1.0, 2.0]]);
        a.zip_apply(&matrix(&[&[3.0, 4.0]]), |x, y| x * y);
        assert_eq!(a, matrix(&[&[3.0, 8.0]]));
    }

    #[test]
    #[should_panic(expected = "zip_apply: 1 × 2 matrix and 2 × 1 matrix")]
    fn zip_apply_panics_on_mismatched_shapes() {
        Matrix::zeros(1, 2).zip_apply(&Matrix::zeros(2, 1), |x, _| x);
    }

    #[test]
    fn apply_and_slice_rows() {
        let mut a = matrix(&[&[1.0], &[2.0], &[3.0]]);
        a.apply(|x| x * 2.0);
        assert_eq!(a.slice_rows(1, 3), matrix(&[&[4.0], &[6.0]]));
    }

    #[test]
    #[should_panic(expected = "slice_rows: rows 2..4 are outside 3 × 1 matrix")]
    fn slice_rows_panics_outside_the_matrix() {
        Matrix::zeros(3, 1).slice_rows(2, 4);
    }

    #[test]
    fn select_rows_copies_rows_in_the_given_order() {
        let a = matrix(&[&[1.0], &[2.0], &[3.0]]);
        assert_eq!(a.select_rows(&[2, 0, 2]), matrix(&[&[3.0], &[1.0], &[3.0]]));
    }

    #[test]
    #[should_panic(expected = "select_rows: row 3 is outside 3 × 1 matrix")]
    fn select_rows_panics_outside_the_matrix() {
        Matrix::zeros(3, 1).select_rows(&[0, 3]);
    }

    #[test]
    fn results_are_stored_row_after_row() {
        let a = matrix(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let b = matrix(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
        assert_eq!(a.transpose_matmul(&b).data(), &[6.0, 8.0, 8.0, 10.0]);
        assert_eq!(
            a.matmul_transpose(&a).data(),
            &[5.0, 11.0, 17.0, 11.0, 25.0, 39.0, 17.0, 39.0, 61.0]
        );
        assert_eq!(a.select_rows(&[2, 0]).data(), &[5.0, 6.0, 1.0, 2.0]);
        assert_eq!(a.sum_rows().data(), &[9.0, 12.0]);
    }

    #[test]
    fn wrap_stores_column_major_arrays_row_after_row() {
        let rows = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let column_major = rows.t().to_owned();
        assert!(!column_major.is_standard_layout());
        let wrapped = Matrix::wrap(column_major);
        assert_eq!(wrapped.data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        assert_eq!(wrapped.row(1), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn argmax_rows_keeps_the_first_of_equal_values() {
        let a = matrix(&[&[0.1, 0.7, 0.2], &[5.0, 5.0, 1.0], &[-3.0, -2.0, -1.0]]);
        assert_eq!(a.argmax_rows(), vec![1, 0, 2]);
    }
}

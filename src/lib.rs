#![feature(iter_arith)]
#![feature(test)]
#![recursion_limit="256"]

extern crate test;

use std::iter;

use std::ops::Mul;

#[derive(PartialEq, Clone, Debug)]
pub struct Matrix {
    num_rows: usize,
    num_cols: usize,
    data: Vec<f64>,
}

impl Matrix
{
    pub fn get(&self, row: usize, col: usize) -> &f64 {
        assert!(col < self.num_cols);
        assert!(row < self.num_rows);
        unsafe {
            self.get_unchecked(row, col)
        }
    }

    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &f64 {
        self.data.get_unchecked(col + self.num_cols * row)
    }

    pub unsafe fn from_rows_unsafe<Rows, Row>(num_rows: usize, num_cols: usize, rows: Rows) -> Self
        where Rows: IntoIterator<Item = Row>,
              Row: IntoIterator<Item = f64>
    {
        // Benchmarks show that repeatedly extending a mutable vector is much
        // faster than using `flat_map` on `rows`, for matrices of all sizes.
        let mut data = Vec::with_capacity(num_rows * num_cols);

        for row in rows.into_iter().take(num_rows) {
            data.extend(row.into_iter().take(num_cols));
        }

        Matrix {
            num_rows: num_rows,
            num_cols: num_cols,
            data: data,
        }
    }

    pub fn tabulate<F>(num_rows: usize, num_cols: usize, f: F) -> Self
        where F: Fn(usize, usize) -> f64
    {

        let nrows = num_rows;
        let ncols = num_cols;

        unsafe {
            Self::from_rows_unsafe(num_rows,
                                   num_cols,
                                   (0..nrows).map(|row| {
                                       (0..ncols)
                                           .zip(iter::repeat(row))
                                           .map(|(col, row)| f(row, col))
                                   }))
        }
    }

    pub fn iter_rows(&self) -> std::slice::Chunks<f64> {
        self.data.chunks(self.num_cols)
    }

    pub fn iter_cols(&self) -> IterMatrixCols {
        IterMatrixCols {
            current_col: 0,
            matrix: self,
        }
    }
}

pub struct IterMatrixCols<'a>
{
    current_col: usize,
    matrix: &'a Matrix,
}

pub struct IterMatrixCol<'a>
{
    current_col: usize,
    current_row: usize,
    matrix: &'a Matrix,
}

impl<'a> Iterator for IterMatrixCols<'a>
{
    type Item = IterMatrixCol<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col < self.matrix.num_cols {
            self.current_col += 1;
            Some(IterMatrixCol {
                current_col: self.current_col - 1,
                current_row: 0,
                matrix: self.matrix,
            })
        } else {
            None
        }
    }
}

impl<'a> Iterator for IterMatrixCol<'a>
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.matrix.num_rows {
            self.current_row += 1;
            Some(self.matrix.get(self.current_row - 1, self.current_col))
        } else {
            None
        }
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix
{
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Matrix {
        unsafe {
            Matrix::from_rows_unsafe(self.num_rows,
                                     rhs.num_cols,
                                     self.iter_rows().map(|lhs_row| {
                rhs.iter_cols()
                    .zip(iter::repeat(lhs_row))
                    .map(|(rhs_col, lhs_row)| lhs_row.iter().zip(rhs_col).map(|(x, y)| x * y).sum())
            }))
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use test::Bencher;

    #[bench]
    fn matrix_mul(bencher: &mut Bencher) {
        let s1 = 100;
        let s2 = 100;
        let s3 = 100;

        let mat1 = Matrix::tabulate(s1, s2, |row, col| (row * col) as f64);
        let mat2 = Matrix::tabulate(s2, s3, |row, col| (row + col) as f64);

        bencher.iter(|| {
            &mat1 * &mat2
        })
    }
}

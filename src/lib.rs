#![feature(zero_one, iter_arith)]
#![feature(test)]
#![recursion_limit="256"]

//! Matrices with well-typed multiplication.

extern crate test;

pub mod type_nats;

use type_nats::*;

use std::iter;
use std::ptr;

use std::ops::Mul;
use std::ops::Add;

use std::marker::PhantomData;

/// An `N` by `M` matrix.
///
/// `N` and `M` should be either type-level natural numbers, or else some type
/// which implements `AsNat` that acts as a placeholder for a static size, and
/// whose `as_nat()` function returns the actual (possibly dynamic) size.
///
/// `N` is number of rows, `M` is number of columns.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Matrix<N, M, T> {
    num_rows: N,
    num_cols: M,
    data: Vec<T>,
}

/// A column vector can be represented as an `n x 1` matrix.
pub type Vector<N, T> = Matrix<N, N1, T>;

impl<N, M, T> Matrix<N, M, T>
    where N: AsNat + Clone,
          M: AsNat + Clone
{
    // =============================================================================
    // Basic functionality
    // =============================================================================

    /// Get the runtime length of a row (or equivalently number of columns) in this matrix.
    #[inline(always)]
    pub fn row_len(&self) -> usize {
        self.num_cols.as_nat()
    }

    /// Get the runtime length of a column (or equivalently number of rows) in this matrix.
    #[inline(always)]
    pub fn col_len(&self) -> usize {
        self.num_rows.as_nat()
    }

    #[inline(always)]
    fn row_col_index(&self, row: usize, col: usize) -> usize {
        col + self.row_len() * row
    }

    /// Gets a reference to the value at the given row and column.
    ///
    /// # Panics
    /// When the row or column is out of bounds.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> &T {
        assert!(col < self.num_cols.as_nat());
        assert!(row < self.num_rows.as_nat());
        unsafe {
            self.get_unchecked(row, col)
        }
    }

    /// Gets a reference to the value at the given row and column.
    ///
    /// # Safety
    /// The row and column must be in bounds.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T {
        self.data.get_unchecked(self.row_col_index(row, col))
    }

    /// Gets a mutable reference to the value at the given row and column.
    ///
    /// # Panics
    /// When the row or column is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        assert!(col < self.num_cols.as_nat());
        assert!(row < self.num_rows.as_nat());
        let idx = self.row_col_index(row, col);
        unsafe {
            self.data.get_unchecked_mut(idx)
        }
    }

    /// Modify each cell in the matrix using some function of the previous value
    /// and its row and column.
    pub fn modify_each<F>(&mut self, f: F)
        where F: Fn(&mut T, usize, usize) -> T
    {
        for (rowi, row) in self.iter_rows_mut().enumerate() {
            for (coli, x) in row.iter_mut().enumerate() {
                *x = f(x, rowi, coli);
            }
        }
    }

    /// Return a new matrix with each cell the result of applying the given
    /// function to the original cell in its position.
    pub fn map<F, U>(&self, f: F) -> Matrix<N, M, U>
        where F: Fn(&T) -> U
    {
        unsafe {
            Matrix::from_rows_unsafe(self.num_rows.clone(),
                                     self.num_cols.clone(),
                                     self.iter_rows().map(|i| i.iter().map(|ref x| f(x))))
        }
    }

    /// Transpose the matrix by moving values into the new matrix (thus
    /// consuming the old matrix).
    pub fn transpose(self) -> Matrix<M, N, T> {
        let num_rows = self.num_rows.clone();
        let num_cols = self.num_cols.clone();
        unsafe { Matrix::from_rows_unsafe(num_cols, num_rows, self.into_cols()) }
    }

    /// Transpose the matrix by referencing cells in the original matrix.
    ///
    /// # Examples
    ///
    /// With copy types, it is often useful to follow this by mapping a derefence:
    ///
    /// ```
    /// use matrix::*;
    /// use matrix::type_nats::*;
    ///
    /// let vals = vec![vec![1, 2], vec![3, 4]];
    /// let mat: Matrix<N2, N2, i32> = Matrix::from_rows_auto(vals.clone()).unwrap();
    /// let transpose = mat.transposed().map(|&x| *x);
    ///
    /// assert_eq!(transpose, Matrix::from_cols_auto(vals).unwrap());
    /// ```
    pub fn transposed(&self) -> Matrix<M, N, &T> {
        unsafe {
            Matrix::from_rows_unsafe(self.num_cols.clone(),
                                     self.num_rows.clone(),
                                     self.iter_cols())
        }
    }

    // =============================================================================
    // Creation
    // =============================================================================

    /// Efficiently creates a matrix with the given rows.
    ///
    /// # Safety
    ///
    /// Neither `num_rows` nor `num_cols` may represent `0`.
    ///
    /// The length of each row in the provided iterator must match
    /// `num_cols.as_nat()`, and the number of rows in the provided iterator
    /// must match `num_rows.as_nat()`.
    pub unsafe fn from_rows_unsafe<Rows, Row>(num_rows: N, num_cols: M, rows: Rows) -> Self
        where Rows: IntoIterator<Item = Row>,
              Row: IntoIterator<Item = T>
    {
        // Benchmarks show that repeatedly extending a mutable vector is much
        // faster than using `flat_map` on `rows`, for matrices of all sizes.
        let mut data = Vec::with_capacity(num_rows.as_nat() * num_cols.as_nat());

        for row in rows.into_iter().take(num_rows.as_nat()) {
            data.extend(row.into_iter().take(num_cols.as_nat()));
        }

        Matrix {
            num_rows: num_rows,
            num_cols: num_cols,
            data: data,
        }
    }

    /// Efficiently creates a matrix with the given rows.
    ///
    /// Returns `None` if any of the row lengths are too short or not enough
    /// rows are provided, or if either of `num_rows` or `num_cols` represents
    /// `0`.
    pub fn from_rows<Rows, Row>(num_rows: N, num_cols: M, rows: Rows) -> Option<Self>
        where Rows: IntoIterator<Item = Row>,
              Row: IntoIterator<Item = T>
    {
        if num_rows.as_nat() == 0 || num_cols.as_nat() == 0 {
            return None;
        }

        let target_size = num_rows.as_nat() * num_cols.as_nat();

        let res = unsafe { Self::from_rows_unsafe(num_rows, num_cols, rows) };

        // check that we have actually filled the matrix
        if res.data.len() != target_size {
            return None;
        }

        Some(res)
    }

    /// Creates a matrix with the given columns.
    ///
    /// Returns `None` if any of the column lengths are too short or not enough
    /// columns are provided, or either of `num_rows` or `num_cols` represents
    /// `0`.
    pub fn from_cols<Cols, Col>(num_rows: N, num_cols: M, cols: Cols) -> Option<Self>
        where Cols: IntoIterator<Item = Col>,
              Col: IntoIterator<Item = T>
    {
        Matrix::from_rows(num_cols, num_rows, cols).map(Matrix::transpose)
    }

    /// Creates a matrix by the tabulating the given function over row and
    /// column indices.
    pub fn tabulate<F>(num_rows: N, num_cols: M, f: F) -> Self
        where F: Fn(usize, usize) -> T
    {

        let nrows = num_rows.as_nat();
        let ncols = num_cols.as_nat();

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

    /// Creates a matrix by the tabulating the given function over row and
    /// column indices. Sizes are implicitly given by the types.
    pub fn tabulate_auto<F>(f: F) -> Self
        where F: Fn(usize, usize) -> T,
              N: TypeNat + Clone,
              M: TypeNat + Clone
    {
        Self::tabulate(N::get_sing(), M::get_sing(), f)
    }

    /// Efficiently creates a matrix with the given rows, where the sizes are
    /// implicitly given by the types.
    ///
    /// Returns `None` if any of the row lengths are too short or not enough
    /// rows are provided.
    pub fn from_rows_auto<Rows, Row>(elems: Rows) -> Option<Self>
        where Rows: IntoIterator<Item = Row>,
              Row: IntoIterator<Item = T>,
              N: TypeNat,
              M: TypeNat
    {
        Self::from_rows(N::get_sing(), M::get_sing(), elems)
    }

    /// Creates a matrix with the given columns, where the sizes are implicitly
    /// given by the types.
    ///
    /// Returns `None` if any of the column lengths are too short or not enough
    /// columns are provided.
    pub fn from_cols_auto<Cols, Col>(elems: Cols) -> Option<Self>
        where Cols: IntoIterator<Item = Col>,
              Col: IntoIterator<Item = T>,
              N: TypeNat,
              M: TypeNat
    {
        Self::from_cols(N::get_sing(), M::get_sing(), elems)
    }

    /// Create a matrix from the given iterator of rows, padding out each row with
    /// the type's default value if it is not long enough. Also pad out the row
    /// iterator with empty (default valued) rows if it is not long enough.
    pub fn from_rows_padded<Rows, Row>(num_rows: N, num_cols: M, elems: Rows) -> Self
        where Rows: IntoIterator<Item = Row>,
              Row: IntoIterator<Item = T>,
              T: Default + Clone
    {
        let nrows = num_rows.as_nat();
        let ncols = num_cols.as_nat();
        unsafe {
            let mut res = Self::from_rows_unsafe(num_rows,
                                                 num_cols,
                                                 elems.into_iter()
                                                     .map(|v| {
                                                         v.into_iter()
                                                             .chain(iter::repeat(T::default()))
                                                             .take(ncols)
                                                     }));

            if res.data.len() < nrows * ncols {
                let extra = iter::repeat(T::default()).take(nrows * ncols - res.data.len());
                res.data.extend(extra);
            }

            res
        }
    }

    /// Create a matrix from the given iterator of rows, padding out each row with
    /// the type's default value if it is not long enough. Also pad out the row
    /// iterator with empty (default valued) rows if it is not long enough.
    pub fn from_rows_padded_auto<Rows, Row>(elems: Rows) -> Self
        where Rows: IntoIterator<Item = Row>,
              Row: IntoIterator<Item = T>,
              T: Default + Clone,
              N: TypeNat,
              M: TypeNat
    {
        Self::from_rows_padded(N::get_sing(), M::get_sing(), elems)
    }

    // =============================================================================
    // Row Iterators
    // =============================================================================

    /// Iterates over the rows in the matrix as slices.
    #[inline]
    pub fn iter_rows(&self) -> std::slice::Chunks<T> {
        self.data.chunks(self.row_len())
    }

    /// Mutably iterates over the rows in the matrix as slices.
    #[inline]
    pub fn iter_rows_mut(&mut self) -> std::slice::ChunksMut<T> {
        let row_len = self.row_len();
        self.data.chunks_mut(row_len)
    }

    /// Iterates over and consume each of the rows in the matrix. Each row is
    /// provided as its own iterator.
    #[inline]
    pub fn into_rows(self) -> IntoMatrixRows<N, M, T> {
        let data = IntoMatrixData::new(self);
        let end_ptr = unsafe {
            data.start_ptr.offset((data.num_rows.as_nat() * data.num_cols.as_nat()) as isize)
        };

        IntoMatrixRows {
            data: data,
            end_ptr: end_ptr,
        }
    }

    // =============================================================================
    // Column Iterators
    // =============================================================================

    /// Iterates over the columns in a matrix, providing each column as its own
    /// iterator.
    pub fn iter_cols(&self) -> IterMatrixCols<N, M, T> {
        IterMatrixCols {
            current_col: 0,
            matrix: self,
        }
    }

    /// Mutably iterates over the columns in a matrix, providing each column as
    /// its own iterator.
    pub fn iter_cols_mut(&mut self) -> IterMatrixColsMut<N, M, T> {
        let start_row_top = self.data.as_mut_ptr();
        let end_row_top = unsafe { start_row_top.offset(self.row_len() as isize) };

        IterMatrixColsMut {
            _marker: PhantomData,
            start_row_top: start_row_top,
            end_row_top: end_row_top,
            num_rows: self.num_rows.clone(),
            num_cols: self.num_cols.clone(),
        }
    }

    /// Iterates over and consume the columns in a matrix, providing each column
    /// as its own iterator.
    pub fn into_cols(self) -> IntoMatrixCols<N, M, T> {
        let data = IntoMatrixData::new(self);
        let end_ptr = unsafe { data.start_ptr.offset(data.num_cols.as_nat() as isize) };

        IntoMatrixCols {
            data: data,
            end_ptr: end_ptr,
        }
    }
}

// =============================================================================
// Iterator structs
// =============================================================================

struct IntoMatrixData<N, M, T> {
    num_rows: N,
    num_cols: M,
    // keep hold of a vector purely so the memory is deallocated properly once
    // we're done iterating
    _data: Vec<T>,
    start_ptr: *const T,
}

impl<N, M, T> IntoMatrixData<N, M, T> {
    fn new(mut mat: Matrix<N, M, T>) -> IntoMatrixData<N, M, T> {
        let start_ptr = mat.data.as_mut_ptr();

        // Store the underlying vector, but set its length to 0. Ensures that
        // the memory will be deallocated, but contents will not be dropped more
        // than once.
        unsafe {
            mat.data.set_len(0);
        }

        IntoMatrixData {
            num_rows: mat.num_rows,
            num_cols: mat.num_cols,
            _data: mat.data,
            start_ptr: start_ptr,
        }
    }
}

/// Iterator over matrix rows.
pub struct IntoMatrixRows<N, M, T> {
    data: IntoMatrixData<N, M, T>,
    end_ptr: *const T,
}

/// Iterator over elements in a matrix row.
pub struct IntoMatrixRow<N, M, T> {
    _marker: PhantomData<(N, M)>,
    start_ptr: *const T,
    end_ptr: *const T,
}

impl<N, M, T> Iterator for IntoMatrixRows<N, M, T>
    where M: AsNat + Clone
{
    type Item = IntoMatrixRow<N, M, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.start_ptr != self.end_ptr {
            let new_start =
                unsafe { self.data.start_ptr.offset(self.data.num_cols.as_nat() as isize) };
            let res = IntoMatrixRow {
                _marker: PhantomData,
                start_ptr: self.data.start_ptr,
                end_ptr: new_start,
            };

            self.data.start_ptr = new_start;
            Some(res)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end_ptr as usize - self.data.start_ptr as usize) / self.data.num_cols.as_nat();
        (len, Some(len))
    }
}

impl<N, M, T> Iterator for IntoMatrixRow<N, M, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_ptr != self.end_ptr {
            let res = unsafe { ptr::read(self.start_ptr) };
            unsafe {
                self.start_ptr = self.start_ptr.offset(1);
            }

            Some(res)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end_ptr as usize - self.start_ptr as usize;
        (len, Some(len))
    }
}

/// Iterator over borrowed matrix columns.
pub struct IterMatrixCols<'a, N, M, T>
    where T: 'a,
          N: 'a,
          M: 'a
{
    current_col: usize,
    matrix: &'a Matrix<N, M, T>,
}

/// Iterator over borrowed elements in a matrix column.
pub struct IterMatrixCol<'a, N, M, T>
    where T: 'a,
          N: 'a,
          M: 'a
{
    current_col: usize,
    current_row: usize,
    matrix: &'a Matrix<N, M, T>,
}

impl<'a, N, M, T> Iterator for IterMatrixCols<'a, N, M, T>
    where N: AsNat + Clone,
          M: AsNat + Clone
{
    type Item = IterMatrixCol<'a, N, M, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col < self.matrix.num_cols.as_nat() {
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

    // #[inline]
    // fn size_hint(&self) -> (usize, Option<usize>) {
    //     let len = self.matrix.num_cols.as_nat() - self.current_col;
    //     (len, Some(len))
    // }
}

impl<'a, N, M, T> Iterator for IterMatrixCol<'a, N, M, T>
    where N: AsNat + Clone,
          M: AsNat + Clone
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.matrix.num_rows.as_nat() {
            self.current_row += 1;
            Some(self.matrix.get(self.current_row - 1, self.current_col))
        } else {
            None
        }
    }

    // #[inline]
    // fn size_hint(&self) -> (usize, Option<usize>) {
    //     let len = self.matrix.num_rows.as_nat() - self.current_row;
    //     (len, Some(len))
    // }
}

/// Iterator over mutably borrowed matrix columns.
pub struct IterMatrixColsMut<'a, N, M, T>
    where T: 'a,
          N: 'a,
          M: 'a
{
    _marker: PhantomData<&'a mut Matrix<N, M, T>>,
    start_row_top: *mut T,
    end_row_top: *mut T,
    num_rows: N,
    num_cols: M,
}

/// Iterator over mutably borrowed elements in a matrix column.
pub struct IterMatrixColMut<'a, N, M, T>
    where T: 'a,
          N: 'a,
          M: 'a
{
    _marker: PhantomData<&'a mut Matrix<N, M, T>>,
    start_ptr: *mut T,
    end_ptr: *mut T,
    num_cols: M,
}

impl<'a, N, M, T> Iterator for IterMatrixColsMut<'a, N, M, T>
    where N: AsNat + Clone,
          M: AsNat + Clone
{
    type Item = IterMatrixColMut<'a, N, M, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_row_top != self.end_row_top {
            let res = IterMatrixColMut {
                _marker: PhantomData,
                start_ptr: self.start_row_top,
                end_ptr: unsafe {
                    self.start_row_top
                        .offset((self.num_cols.as_nat() * self.num_rows.as_nat()) as isize)
                },
                num_cols: self.num_cols.clone(),
            };

            unsafe {
                self.start_row_top = self.start_row_top.offset(1);
            }

            Some(res)
        } else {
            None
        }
    }
}

impl<'a, N, M, T> Iterator for IterMatrixColMut<'a, N, M, T>
    where M: AsNat
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_ptr != self.end_ptr {
            let res = unsafe { self.start_ptr.as_mut().unwrap() };

            unsafe {
                self.start_ptr = self.start_ptr.offset(self.num_cols.as_nat() as isize);
            }

            Some(res)
        } else {
            None
        }
    }
}

/// Iterator over matrix columns.
pub struct IntoMatrixCols<N, M, T> {
    data: IntoMatrixData<N, M, T>,
    end_ptr: *const T,
}

/// Iterator over elements in a matrix column.
pub struct IntoMatrixCol<N, M, T> {
    _marker: PhantomData<N>,
    num_cols: M,
    start_ptr: *const T,
    end_ptr: *const T,
}

impl<N, M, T> Iterator for IntoMatrixCols<N, M, T>
    where N: AsNat + Clone,
          M: AsNat + Clone
{
    type Item = IntoMatrixCol<N, M, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.start_ptr != self.end_ptr {
            let res = IntoMatrixCol {
                _marker: PhantomData,
                start_ptr: self.data.start_ptr,
                end_ptr: unsafe {
                    self.data
                        .start_ptr
                        .offset((self.data.num_cols.as_nat() *
                                 self.data
                            .num_rows
                            .as_nat()) as isize)
                },
                num_cols: self.data.num_cols.clone(),
            };

            unsafe {
                self.data.start_ptr = self.data.start_ptr.offset(1);
            }

            Some(res)
        } else {
            None
        }
    }
}

impl<N, M, T> DoubleEndedIterator for IntoMatrixCols<N, M, T>
    where N: AsNat + Clone,
          M: AsNat + Clone
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.data.start_ptr != self.end_ptr {
            unsafe {
                self.end_ptr = self.end_ptr.offset(-1);
            }
            Some(IntoMatrixCol {
                _marker: PhantomData,
                start_ptr: self.end_ptr,
                end_ptr: unsafe {
                    self.end_ptr
                        .offset((self.data.num_cols.as_nat() *
                                 self.data
                            .num_rows
                            .as_nat()) as isize)
                },
                num_cols: self.data.num_cols.clone(),
            })
        } else {
            None
        }
    }
}

impl<N, M, T> Iterator for IntoMatrixCol<N, M, T>
    where M: AsNat
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_ptr != self.end_ptr {
            let res = unsafe { ptr::read(self.start_ptr) };

            unsafe {
                self.start_ptr = self.start_ptr.offset(self.num_cols.as_nat() as isize);
            }

            Some(res)
        } else {
            None
        }
    }
}

impl<N, M, T> DoubleEndedIterator for IntoMatrixCol<N, M, T>
    where M: AsNat
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start_ptr != self.end_ptr {
            unsafe {
                self.start_ptr = self.start_ptr.offset(self.num_cols.as_nat() as isize);

                Some(ptr::read(self.end_ptr))
            }
        } else {
            None
        }
    }
}

// =============================================================================
//  Special cases
// =============================================================================

impl<N, T> Matrix<N, N, T>
    where N: AsNat + Clone,
          T: std::num::Zero + std::num::One
{
    /// Creates an identity matrix with the given size.
    pub fn identity(size: N) -> Self {
        Self::tabulate(size.clone(), size, |row, col| if row == col {
            T::one()
        } else {
            T::zero()
        })
    }

    /// Creates an identity matrix with the size implicitly given by the type.
    pub fn identity_auto() -> Self
        where N: TypeNat
    {
        Self::identity(N::get_sing())
    }

    pub fn transpose_inplace(&mut self) {
        for col in 0..self.num_cols.as_nat() {
            for row in (col + 1)..self.num_rows.as_nat() {
                let idx_1 = self.row_col_index(row, col);
                let idx_2 = self.row_col_index(col, row);
                self.data.swap(idx_1, idx_2);
            }
        }
    }
}

// =============================================================================
//  Arithmetic
// =============================================================================

impl<'a, 'b, N, M, O, T1, T2> Mul<&'b Matrix<M, O, T2>> for &'a Matrix<N, M, T1>
    where N: AsNat + Clone,
          M: AsNat + Clone,
          O: AsNat + Clone,
          &'a T1: Mul<&'b T2, Output = T1>,
          T1: Add<T1, Output = T1> + std::num::Zero
{
    type Output = Matrix<N, O, T1>;

    fn mul(self, rhs: &'b Matrix<M, O, T2>) -> Matrix<N, O, T1> {
        unsafe {
            Matrix::from_rows_unsafe(self.num_rows.clone(),
                                     rhs.num_cols.clone(),
                                     self.iter_rows().map(|lhs_row| {
                rhs.iter_cols()
                    .zip(iter::repeat(lhs_row))
                    .map(|(rhs_col, lhs_row)| lhs_row.iter().zip(rhs_col).map(|(x, y)| x * y).sum())
            }))
        }
    }
}

impl<'a, N, M, O, T1, T2> Mul<&'a Matrix<M, O, T2>> for Matrix<N, M, T1>
    where N: AsNat + Clone + 'a,
          M: AsNat + Clone,
          O: AsNat + Clone,
          for<'b> &'b T1: Mul<&'a T2, Output = T1>,
          T1: Add<T1, Output = T1> + std::num::Zero + 'a
{
    type Output = Matrix<N, O, T1>;

    fn mul(self, rhs: &'a Matrix<M, O, T2>) -> Matrix<N, O, T1> {
        &self * rhs
    }
}

impl<N, M, O, T1, T2> Mul<Matrix<M, O, T2>> for Matrix<N, M, T1>
    where N: AsNat + Clone,
          M: AsNat + Clone,
          O: AsNat + Clone,
          for<'a, 'b> &'a T1: Mul<&'b T2, Output = T1>,
          T1: Add<T1, Output = T1> + std::num::Zero
{
    type Output = Matrix<N, O, T1>;

    fn mul(self, rhs: Matrix<M, O, T2>) -> Matrix<N, O, T1> {
        self * &rhs
    }
}

impl<N, T1> Matrix<N, N1, T1>
    where N: AsNat + Clone
{
    /// Treating `n x 1` matrices as column vectors, take the dot product.
    pub fn dot<'a, 'b, T2>(&'a self, other: &'b Matrix<N, N1, T2>) -> T1
        where for<'c> &'c T1: Mul<&'b T2, Output = T1>,
              T1: Add<T1, Output = T1> + std::num::Zero,
              T1: Clone + 'b
    {
        let tp: Matrix<_, _, T1> = self.transposed().map(|x| (*x).clone());
        (tp * other).get(0, 0).clone()
    }

    /// Treating `n x 1` matrices as column vectors, take the dot product.
    pub fn dot_owned<T2>(self, other: Matrix<N, N1, T2>) -> T1
        where for<'a, 'b> &'a T1: Mul<&'b T2, Output = T1>,
              T1: Add<T1, Output = T1> + std::num::Zero
    {
        let tp = self.transpose();
        (tp * other).data.into_iter().next().unwrap()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::type_nats::*;

    #[test]
    fn from_rows() {
        // valid
        let success: Option<Matrix<N2, N3, i32>> = Matrix::from_rows_auto(vec![vec![0, 1, 2],
                                                                               vec![2, 3, 4]]);

        // wrong number of rows
        let fail1: Option<Matrix<N2, N3, i32>> = Matrix::from_rows_auto(vec![vec![0, 1, 2]]);

        // wrong number of columns
        let fail2: Option<Matrix<N2, N3, i32>> = Matrix::from_rows_auto(vec![vec![0, 1, 2],
                                                                             vec![2, 3]]);

        assert_eq!(success,
                   Some(Matrix {
                       num_rows: N2::get_sing(),
                       num_cols: N3::get_sing(),
                       data: vec![0, 1, 2, 2, 3, 4],
                   }));

        assert_eq!(fail1, None);
        assert_eq!(fail2, None);
    }

    #[test]
    fn into_rows() {
        let real_rows = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let mat: Matrix<N2, N3, i32> = Matrix::from_rows_auto(real_rows.clone()).unwrap();

        let mut elems: Vec<_> = Vec::new();

        for row in mat.into_rows() {
            let mut row_elems: Vec<_> = Vec::new();

            for x in row {
                row_elems.push(x);
            }

            elems.push(row_elems);
        }

        assert_eq!(elems, real_rows)
    }

    #[test]
    fn into_cols() {
        let mat: Matrix<N2, N3, i32> = Matrix::from_rows_auto(vec![vec![0, 1, 2], vec![2, 3, 4]])
            .unwrap();

        let real_cols = vec![vec![0, 2], vec![1, 3], vec![2, 4]];
        let mut elems: Vec<_> = Vec::new();

        for col in mat.into_cols() {
            let mut col_elems: Vec<_> = Vec::new();

            for x in col {
                col_elems.push(x);
            }

            elems.push(col_elems);
        }

        assert_eq!(elems, real_cols)
    }

    #[test]
    fn tranposition() {
        let original: Matrix<N2, N3, i32> =
            Matrix::from_rows_auto(vec![vec![0, 1, 2], vec![2, 3, 4]]).unwrap();

        let target: Matrix<N3, N2, i32> =
            Matrix::from_rows_auto(vec![vec![0, 2], vec![1, 3], vec![2, 4]]).unwrap();

        assert_eq!(target, original.transpose());
    }

    #[test]
    fn dot() {
        let v1: Vector<N4, i32> = Matrix::from_cols_auto(vec![vec![1, 2, 3, 4]]).unwrap();
        let v2: Vector<N4, i32> = Matrix::from_cols_auto(vec![vec![5, 6, 7, 8]]).unwrap();
        let dp: i32 = v1.dot(&v2);

        assert_eq!(dp, 5 + 12 + 21 + 32);
    }

    #[test]
    fn iter_cols() {
        let vals = vec![vec![1, 4], vec![2, 5], vec![3, 6]];
        let mat: Matrix<N2, N3, i32> = Matrix::from_cols_auto(vals.clone()).unwrap();

        let recovered: Vec<Vec<i32>> = mat.iter_cols().map(|v| v.map(|i| *i).collect()).collect();

        assert_eq!(recovered, vals);
    }

    #[test]
    fn matrix_mul() {
        let lhs: Matrix<N3, N2, i32> =
            Matrix::from_rows_auto(vec![vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        let rhs: Matrix<N2, N3, i32> = Matrix::from_rows_auto(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .unwrap();

        let correct_result: Matrix<N3, N3, i32> = Matrix::from_rows_auto(vec![
                vec![9, 12, 15],
                vec![19, 26, 33],
                vec![29, 40, 51],
            ])
            .unwrap();

        assert_eq!(lhs * rhs, correct_result);
    }

    #[test]
    fn mul_identity() {
        let mat: Matrix<N3, N3, i32> =
            Matrix::from_rows_auto(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]).unwrap();

        let id: Matrix<N3, N3, i32> = Matrix::identity_auto();

        assert_eq!(mat.clone() * id, mat);
    }

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum A {}

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum B {}

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum C {}

    #[test]
    fn dynamic_size_mul() {
        let t_a = Tagged::new(3);
        let t_b = Tagged::new(2);
        let t_c = Tagged::new(3);

        let lhs: Matrix<TaggedNat<A>, TaggedNat<B>, i32> =
            Matrix::from_rows(t_a, t_b, vec![vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        let rhs: Matrix<TaggedNat<B>, TaggedNat<C>, i32> =
            Matrix::from_rows(t_b, t_c, vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

        let correct_result: Matrix<TaggedNat<A>, TaggedNat<C>, i32> = Matrix::from_rows(t_a,
                                                                                        t_c,
                                                                                        vec![
                    vec![9, 12, 15],
                    vec![19, 26, 33],
                    vec![29, 40, 51],
                ])
            .unwrap();

        assert_eq!(&lhs * &rhs, correct_result);
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use super::type_nats::*;
    use test::Bencher;

    #[bench]
    fn tabulate(bencher: &mut Bencher) {
        bencher.iter(|| {
            Matrix::<N10, N10, usize>::tabulate_auto(|row, col| row * col)
        })
    }

    #[bench]
    fn matrix_mul(bencher: &mut Bencher) {
        let mat1 = Matrix::<N100, N100, usize>::tabulate_auto(|row, col| row * col);
        let mat2 = Matrix::<N100, N100, usize>::tabulate_auto(|row, col| row + col);

        bencher.iter(|| {
            &mat1 * &mat2
        })
    }
}

#![feature(zero_one)]
#![feature(test)]
#![recursion_limit="128"]

extern crate test;

pub mod type_nats;

use type_nats::*;

use std::iter;
use std::iter::FromIterator;
use std::collections::VecDeque;

use std::ops::Mul;
use std::ops::Add;

// =============================================================================
//  Vectors
// =============================================================================

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
pub struct Vector<N, T> {
    size: N,
    vec: Vec<T>,
}

impl<N, T> Vector<N, T>
    where N: AsNat + Copy
{
    #[inline]
    pub fn new_unsafe<It: IntoIterator<Item = T>>(elems: It, size: N) -> Self {
        Vector {
            size: size,
            vec: elems.into_iter().take(size.as_nat()).collect(),
        }
    }

    pub fn new<It: IntoIterator<Item = T>>(elems: It, size: N) -> Option<Self> {
        let res = Self::new_unsafe(elems, size);

        if res.vec.len() == size.as_nat() {
            Some(res)
        } else {
            None
        }
    }
}

impl<N, T> Vector<N, T>
    where N: AsNat + Copy,
          T: Default + Clone
{
    /// Create a new_auto vector from the elements in the given iterator. If there
    /// are too many items in the iterator, truncate. If there are not enough
    /// items in the iterator, pad out with the default value.
    #[inline]
    pub fn new_padded<It: IntoIterator<Item = T>>(elems: It, size: N) -> Self {
        Self::new_unsafe(elems.into_iter().chain(iter::repeat(T::default())), size)
    }
}

impl<N, T> Vector<N, T>
    where N: TypeNat
{
    /// Create a new_auto vector from the elements in the given iterator. If there
    /// are too many items in the iterator, truncate. If there are not enough
    /// items in the iterator, return `None`.
    pub fn new_auto<It: IntoIterator<Item = T>>(elems: It) -> Option<Self> {
        Self::new(elems, N::get_sing())
    }
}

impl<N, T> Vector<N, T>
    where N: TypeNat,
          T: Default + Clone
{
    /// Create a new_auto vector from the elements in the given iterator. If there
    /// are too many items in the iterator, truncate. If there are not enough
    /// items in the iterator, pad out with the default value.
    #[inline]
    pub fn new_padded_auto<It: IntoIterator<Item = T>>(elems: It) -> Self {
        Self::new_padded(elems, N::get_sing())
    }
}

impl<N, T> Vector<N, T> {
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.vec.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.vec.iter_mut()
    }
}

impl<N, T> IntoIterator for Vector<N, T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<'a, N, T> IntoIterator for &'a Vector<N, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<N, T> FromIterator<T> for Vector<N, T>
    where N: TypeNat,
          T: Default + Clone
{
    fn from_iter<I>(iter: I) -> Self
        where I: IntoIterator<Item = T>
    {
        Self::new_padded_auto(iter)
    }
}

impl<N, T1> Vector<N, T1> {
    pub fn dot<'a, 'b, T2: 'b, It>(&'a self, other: It) -> T1
        where It: IntoIterator<Item = &'b T2>,
              &'a T1: Mul<&'b T2, Output = T1>,
              T1: Add<T1, Output = T1> + std::num::Zero
    {
        let mut res = T1::zero();
        let mut rhs_iter = other.into_iter();

        for lhs in self.iter() {
            if let Some(rhs) = rhs_iter.next() {
                res = res + lhs * rhs;
            }
        }

        res
    }
}

// =============================================================================
//  Matrices
// =============================================================================

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
pub struct Matrix<N, M, T> {
    num_cols: M,
    rows: Vector<N, Vector<M, T>>,
}

pub struct IterMatrixCols<N, M, T> {
    num_rows: std::marker::PhantomData<N>,
    num_cols: M,
    current_col: usize,
    remaining_matrix: Vec<VecDeque<T>>,
}

pub struct IterMatrixColsBorrow<'a, N, M, T>
    where T: 'a,
          N: 'a,
          M: 'a
{
    current_col: usize,
    matrix: &'a Matrix<N, M, T>,
}

pub struct IterMatrixColBorrow<'a, N, M, T>
    where T: 'a,
          N: 'a,
          M: 'a
{
    current_col: usize,
    current_row: usize,
    matrix: &'a Matrix<N, M, T>,
}

impl<N, M, T> Iterator for IterMatrixCols<N, M, T>
    where M: AsNat
{
    type Item = <Vector<N, T> as IntoIterator>::IntoIter;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.num_cols.as_nat() {
            return None;
        }

        let next_col: Vec<T> = self.remaining_matrix
            .iter_mut()
            .map(|deque| {
                deque.pop_front().expect("Error: iterating over columns of malformed matrix.")
            })
            .collect();

        self.current_col += 1;

        Some(next_col.into_iter())
    }
}

impl<'a, N, M, T> Iterator for IterMatrixColsBorrow<'a, N, M, T>
    where M: AsNat
{
    type Item = IterMatrixColBorrow<'a, N, M, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col < self.matrix.num_cols.as_nat() {
            self.current_col += 1;
            Some(IterMatrixColBorrow {
                current_col: self.current_col - 1,
                current_row: 0,
                matrix: self.matrix,
            })
        } else {
            None
        }
    }
}

impl<'a, N, M, T> Iterator for IterMatrixColBorrow<'a, N, M, T>
    where N: AsNat + Clone
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.matrix.rows.size.as_nat() {
            self.current_row += 1;
            Some(self.matrix.get(self.current_row - 1, self.current_col))
        } else {
            None
        }
    }
}

impl<N, M, T> Matrix<N, M, T>
    where N: Clone
{
    pub fn into_iter_rows(self) -> <Vector<N, Vector<M, T>> as IntoIterator>::IntoIter {
        self.rows.into_iter()
    }

    pub fn into_iter_cols(self) -> IterMatrixCols<N, M, T> {
        let deques = self.rows.into_iter().map(|r| r.into_iter().collect()).collect();

        IterMatrixCols {
            num_rows: std::marker::PhantomData,
            num_cols: self.num_cols,
            current_col: 0,
            remaining_matrix: deques,
        }
    }

    pub fn iter_rows(&self) -> std::slice::Iter<Vector<M, T>> {
        self.rows.iter()
    }

    pub fn iter_cols(&self) -> IterMatrixColsBorrow<N, M, T> {
        IterMatrixColsBorrow {
            current_col: 0,
            matrix: self,
        }
    }

    pub fn get<'a>(&'a self, row: usize, col: usize) -> &'a T {
        &self.rows.vec[row].vec[col]
    }
}

impl<N, M, T> Matrix<N, M, T>
    where N: AsNat + Copy,
          M: AsNat + Copy
{
    pub fn from_rows_unsafe<It1, It2>(elems: It1, num_rows: N, num_cols: M) -> Self
        where It1: IntoIterator<Item = It2>,
              It2: IntoIterator<Item = T>
    {

        Matrix {
            num_cols: num_cols,
            rows: Vector::new_unsafe(elems.into_iter().map(|v| Vector::new_unsafe(v, num_cols)),
                                     num_rows),
        }
    }

    pub fn from_rows<It1, It2>(elems: It1, num_rows: N, num_cols: M) -> Option<Self>
        where It1: IntoIterator<Item = It2>,
              It2: IntoIterator<Item = T>
    {

        let res = Self::from_rows_unsafe(elems, num_rows, num_cols);

        if res.rows.vec.len() != num_rows.as_nat() {
            return None;
        }

        for v in res.rows.vec.iter() {
            if v.vec.len() != num_cols.as_nat() {
                return None;
            }
        }

        Some(res)
    }

    pub fn transpose(self) -> Matrix<M, N, T> {
        let num_cols = self.num_cols;
        let num_rows = self.rows.size;
        Matrix::from_rows_unsafe(self.into_iter_cols(), num_cols, num_rows)
    }

    pub fn from_cols<It1, It2>(elems: It1, num_rows: N, num_cols: M) -> Option<Self>
        where It1: IntoIterator<Item = It2>,
              It2: IntoIterator<Item = T>
    {
        Matrix::from_rows(elems, num_cols, num_rows).map(Matrix::transpose)
    }
}

impl<N, M, T> Matrix<N, M, T>
    where N: TypeNat,
          M: TypeNat
{
    pub fn from_rows_auto<It1, It2>(elems: It1) -> Option<Self>
        where It1: IntoIterator<Item = It2>,
              It2: IntoIterator<Item = T>
    {
        Self::from_rows(elems, N::get_sing(), M::get_sing())
    }

    pub fn from_cols_auto<It1, It2>(elems: It1) -> Option<Self>
        where It1: IntoIterator<Item = It2>,
              It2: IntoIterator<Item = T>
    {
        Self::from_cols(elems, N::get_sing(), M::get_sing())
    }
}

impl<N, M, T> Matrix<N, M, T>
    where N: AsNat + Copy,
          M: AsNat + Copy,
          T: Default + Clone
{
    pub fn from_rows_padded<It1, It2>(elems: It1, num_rows: N, num_cols: M) -> Self
        where It1: IntoIterator<Item = It2>,
              It2: IntoIterator<Item = T>
    {
        Self::from_rows_unsafe(elems.into_iter()
                                   .chain(iter::empty())
                                   .map(|v| v.into_iter().chain(iter::repeat(T::default()))),
                               num_rows,
                               num_cols)
    }
}

impl<'a, N, M, T1, T2> Mul<&'a Vector<M, T2>> for &'a Matrix<N, M, T1>
    where N: AsNat + Copy,
          &'a T1: Mul<&'a T2, Output = T1>,
          T1: Add<T1, Output = T1> + std::num::Zero
{
    type Output = Vector<N, T1>;

    fn mul(self, rhs: &'a Vector<M, T2>) -> Vector<N, T1> {
        Vector::new_unsafe(self.iter_rows().map(|r| r.dot(rhs)), self.rows.size)
    }
}

impl<'a, N, M, O, T1, T2> Mul<&'a Matrix<M, O, T2>> for &'a Matrix<N, M, T1>
    where N: AsNat + Copy,
          M: AsNat + Copy,
          O: AsNat + Copy,
          &'a T1: Mul<&'a T2, Output = T1>,
          T1: Add<T1, Output = T1> + std::num::Zero
{
    type Output = Matrix<N, O, T1>;

    fn mul(self, rhs: &'a Matrix<M, O, T2>) -> Matrix<N, O, T1> {
        Matrix::from_rows_unsafe(self.iter_rows().map(|lhs_row| {
            let r: Vec<T1> = rhs.iter_cols()
                .map(|rhs_col| lhs_row.dot(rhs_col))
                .collect();

            r
        }),
                                 self.rows.size,
                                 rhs.num_cols)
    }
}

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::type_nats::*;

    #[test]
    fn it_works() {
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
                       num_cols: N3::get_sing(),
                       rows: Vector {
                           size: N2::get_sing(),
                           vec: vec![
                               Vector {
                                   size: N3::get_sing(),
                                   vec: vec![0, 1, 2],
                               },
                               Vector {
                                   size: N3::get_sing(),
                                   vec: vec![2, 3, 4],
                               },],
                       },
                   }));

        assert_eq!(fail1, None);
        assert_eq!(fail2, None);
    }

    #[test]
    fn tranposition() {
        let original: Matrix<N2, N3, i32> =
            Matrix::from_rows_auto(vec![vec![0, 1, 2], vec![2, 3, 4]]).unwrap();

        let target: Matrix<N3, N2, i32> =
            Matrix::from_cols_auto(vec![vec![0, 1, 2], vec![2, 3, 4]]).unwrap();

        assert_eq!(target, original.transpose());
    }

    #[test]
    fn dot() {
        let v1: Vector<N4, i32> = Vector::new_auto(vec![1, 2, 3, 4]).unwrap();
        let v2: Vector<N4, i32> = Vector::new_auto(vec![5, 6, 7, 7]).unwrap();
        let dp: i32 = v1.dot(&v2);

        assert_eq!(dp, 5 + 12 + 21 + 28);
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

        assert_eq!(&lhs * &rhs, correct_result);
    }

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum A {}

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum B {}

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum C {}

    #[test]
    fn dynamic_size_mul() {
        let lhs: Matrix<TaggedNat<A>, TaggedNat<B>, i32> =
            Matrix::from_rows(vec![vec![1, 2], vec![3, 4], vec![5, 6]], Tagged::new(3), Tagged::new(2)).unwrap();
        let rhs: Matrix<TaggedNat<B>, TaggedNat<C>, i32> =
            Matrix::from_rows(vec![vec![1, 2, 3], vec![4, 5, 6]], Tagged::new(2), Tagged::new(3)).unwrap();

        let correct_result: Matrix<TaggedNat<A>, TaggedNat<C>, i32> = Matrix::from_rows(
            vec![
                vec![9, 12, 15],
                vec![19, 26, 33],
                vec![29, 40, 51],
            ], Tagged::new(3), Tagged::new(3)).unwrap();

        assert_eq!(&lhs * &rhs, correct_result);
    }
}

//! Type level (and pseudo-type-level) natural numbers and singletons.

use std::marker::PhantomData;
use std::ops::Deref;

/// A type is `AsNat` if a value can be viewed as a natural number.
pub trait AsNat {
    /// Get the underlying natural number as a `usize`.
    fn as_nat(&self) -> usize;
}

impl AsNat for usize {
    #[inline]
    fn as_nat(&self) -> usize {
        *self
    }
}

/// A type is `Sing` (singleton) if there is exactly one possible value with
/// that type.
pub trait Sing {
    /// Get the singleton value of this type.
    fn get_sing() -> Self;
}

/// `TypeNat` is essentially shorthand for `AsNat + Sing`, with an extra
/// convenience function `get_nat`. This represents a type-level natural number.
pub trait TypeNat: AsNat + Sing {
    /// Get the single natural number associated with this type.
    #[inline]
    fn get_nat() -> usize {
        Self::get_sing().as_nat()
    }
}

impl<T> TypeNat for T where T: AsNat + Sing {}

/// Type-level representation of `0`.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Zero;

/// If `N` is the type-level representation of the number `n`, `Succ<N>` is the
/// type-level representation of `n+1`.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Succ<N> {
    n_marker: PhantomData<N>,
}

/// An arbitrary value tagged (at the type-level only) with some type.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Tagged<Tag, Val> {
    tag_marker: PhantomData<Tag>,
    val: Val,
}

/// `Tagged` specialized with a `usize` value.
pub type TaggedNat<Tag> = Tagged<Tag, usize>;

impl AsNat for Zero {
    #[inline]
    fn as_nat(&self) -> usize {
        0
    }
}

impl Sing for Zero {
    #[inline]
    fn get_sing() -> Self {
        Zero
    }
}

impl<N> AsNat for Succ<N>
    where N: TypeNat
{
    #[inline]
    fn as_nat(&self) -> usize {
        1 + N::get_nat()
    }
}

impl<N> Sing for Succ<N> {
    #[inline]
    fn get_sing() -> Self {
        Succ { n_marker: PhantomData }
    }
}

impl<Tag, Val> Tagged<Tag, Val> {
    #[inline]
    pub fn new(val: Val) -> Self {
        Tagged {
            tag_marker: PhantomData,
            val: val,
        }
    }
}

impl<Tag, Val> Deref for Tagged<Tag, Val> {
    type Target = Val;

    #[inline]
    fn deref(&self) -> &Val {
        &self.val
    }
}

impl<Tag, Val> AsNat for Tagged<Tag, Val>
    where Val: AsNat
{
    #[inline]
    fn as_nat(&self) -> usize {
        self.val.as_nat()
    }
}

impl<Tag, Val> Sing for Tagged<Tag, Val>
    where Val: Sing
{
    #[inline]
    fn get_sing() -> Self {
        Self::new(Val::get_sing())
    }
}

pub type N0 = Zero;
pub type N1 = Succ<N0>;
pub type N2 = Succ<N1>;
pub type N3 = Succ<N2>;
pub type N4 = Succ<N3>;
pub type N5 = Succ<N4>;
pub type N6 = Succ<N5>;
pub type N7 = Succ<N6>;
pub type N8 = Succ<N7>;
pub type N9 = Succ<N8>;
pub type N10 = Succ<N9>;
pub type N11 = Succ<N10>;
pub type N12 = Succ<N11>;
pub type N13 = Succ<N12>;
pub type N14 = Succ<N13>;
pub type N15 = Succ<N14>;
pub type N16 = Succ<N15>;
pub type N17 = Succ<N16>;
pub type N18 = Succ<N17>;
pub type N19 = Succ<N18>;
pub type N20 = Succ<N19>;
pub type N21 = Succ<N20>;
pub type N22 = Succ<N21>;
pub type N23 = Succ<N22>;
pub type N24 = Succ<N23>;
pub type N25 = Succ<N24>;
pub type N26 = Succ<N25>;
pub type N27 = Succ<N26>;
pub type N28 = Succ<N27>;
pub type N29 = Succ<N28>;
pub type N30 = Succ<N29>;
pub type N31 = Succ<N30>;
pub type N32 = Succ<N31>;
pub type N33 = Succ<N32>;
pub type N34 = Succ<N33>;
pub type N35 = Succ<N34>;
pub type N36 = Succ<N35>;
pub type N37 = Succ<N36>;
pub type N38 = Succ<N37>;
pub type N39 = Succ<N38>;
pub type N40 = Succ<N39>;
pub type N41 = Succ<N40>;
pub type N42 = Succ<N41>;
pub type N43 = Succ<N42>;
pub type N44 = Succ<N43>;
pub type N45 = Succ<N44>;
pub type N46 = Succ<N45>;
pub type N47 = Succ<N46>;
pub type N48 = Succ<N47>;
pub type N49 = Succ<N48>;
pub type N50 = Succ<N49>;
pub type N51 = Succ<N50>;
pub type N52 = Succ<N51>;
pub type N53 = Succ<N52>;
pub type N54 = Succ<N53>;
pub type N55 = Succ<N54>;
pub type N56 = Succ<N55>;
pub type N57 = Succ<N56>;
pub type N58 = Succ<N57>;
pub type N59 = Succ<N58>;
pub type N60 = Succ<N59>;
pub type N61 = Succ<N60>;
pub type N62 = Succ<N61>;
pub type N63 = Succ<N62>;
pub type N64 = Succ<N63>;
pub type N65 = Succ<N64>;
pub type N66 = Succ<N65>;
pub type N67 = Succ<N66>;
pub type N68 = Succ<N67>;
pub type N69 = Succ<N68>;
pub type N70 = Succ<N69>;
pub type N71 = Succ<N70>;
pub type N72 = Succ<N71>;
pub type N73 = Succ<N72>;
pub type N74 = Succ<N73>;
pub type N75 = Succ<N74>;
pub type N76 = Succ<N75>;
pub type N77 = Succ<N76>;
pub type N78 = Succ<N77>;
pub type N79 = Succ<N78>;
pub type N80 = Succ<N79>;
pub type N81 = Succ<N80>;
pub type N82 = Succ<N81>;
pub type N83 = Succ<N82>;
pub type N84 = Succ<N83>;
pub type N85 = Succ<N84>;
pub type N86 = Succ<N85>;
pub type N87 = Succ<N86>;
pub type N88 = Succ<N87>;
pub type N89 = Succ<N88>;
pub type N90 = Succ<N89>;
pub type N91 = Succ<N90>;
pub type N92 = Succ<N91>;
pub type N93 = Succ<N92>;
pub type N94 = Succ<N93>;
pub type N95 = Succ<N94>;
pub type N96 = Succ<N95>;
pub type N97 = Succ<N96>;
pub type N98 = Succ<N97>;
pub type N99 = Succ<N98>;
pub type N100 = Succ<N99>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phantom_numbers() {
        assert_eq!(N0::get_nat(), 0);
        assert_eq!(N1::get_nat(), 1);
        assert_eq!(N2::get_nat(), 2);
    }

}

#[cfg(type_nat_benches)]
mod benches {
    use super::*;
    use test::Bencher;

    // if the `bench_as_nat` benchmark runs slower than `bench_9`, something is
    // wrong and the value is being evaluated at run-time.

    #[bench]
    fn bench_as_nat(b: &mut Bencher) {
        b.iter(|| N9::get_nat())
    }

    #[bench]
    fn bench_9(b: &mut Bencher) {
        b.iter(|| 9)
    }
}

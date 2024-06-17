use std::ops::{Div, MulAssign};

use ndarray::{Array1, ArrayView1};
use num_traits::One;

use crate::number::Rational;

pub fn rational_to_z3<'a>(ctx: &'a z3::Context, r: &Rational) -> z3::ast::Real<'a> {
    let (numer, denom) = r.to_integer_ratio();
    let numer = z3::ast::Int::from_str(ctx, &numer.to_string())
        .unwrap()
        .to_real();
    let denom = z3::ast::Int::from_str(ctx, &denom.to_string())
        .unwrap()
        .to_real();
    numer / denom
}

pub fn z3_real_to_rational(real: &z3::ast::Real) -> Option<Rational> {
    if let Some((n, d)) = real.as_real() {
        return Some(Rational::from_frac(n, d));
    }
    let string = real.to_string();
    if let Ok(i) = string.parse::<i64>() {
        Some(Rational::from_int(i))
    } else {
        let words = string.split_whitespace().collect::<Vec<_>>();
        if words.len() == 3 && words[0] == "(/" && words[2].ends_with(')') {
            let n = words[1].parse::<i64>().ok()?;
            let d = words[2][..words[2].len() - 1].parse::<u64>().ok()?;
            return Some(Rational::from_frac(n, d));
        }
        None
    }
}

pub fn rational_to_qepcad(r: &Rational) -> String {
    format!("({r})")
}

pub fn pow<T>(base: T, exp: i32) -> T
where
    T: Clone + One + MulAssign + Div<Output = T>,
{
    if exp == 0 {
        T::one()
    } else if exp < 0 {
        T::one() / pow_nonneg(base, (-exp) as u32)
    } else {
        pow_nonneg(base, exp as u32)
    }
}

pub fn pow_nonneg<T>(base: T, exp: u32) -> T
where
    T: Clone + One + MulAssign,
{
    let mut result = T::one();
    let mut base = base;
    let mut exp = exp;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base.clone();
        }
        base *= base.clone();
        exp /= 2;
    }
    result
}

pub fn norm(vec: &ArrayView1<f64>) -> f64 {
    vec.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn normalize(vec: &ArrayView1<f64>) -> Array1<f64> {
    let norm = norm(vec);
    if norm == 0.0 {
        return vec.to_owned();
    }
    vec.map(|x| x / norm)
}

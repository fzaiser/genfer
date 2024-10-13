use std::ops::{Div, MulAssign};

use ndarray::{Array1, ArrayView1};
use num_traits::{One, Zero};

use crate::numbers::Rational;

pub(crate) fn rational_to_z3<'a>(ctx: &'a z3::Context, r: &Rational) -> z3::ast::Real<'a> {
    let (numer, denom) = r.to_integer_ratio();
    let numer = z3::ast::Int::from_str(ctx, &numer.to_string())
        .unwrap()
        .to_real();
    let denom = z3::ast::Int::from_str(ctx, &denom.to_string())
        .unwrap()
        .to_real();
    numer / denom
}

pub(crate) fn z3_real_to_rational(real: &z3::ast::Real) -> Option<Rational> {
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

pub(crate) fn rational_to_qepcad(r: &Rational) -> String {
    format!("({r})")
}

pub(crate) fn pow<T>(base: T, exp: i32) -> T
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

pub(crate) fn pow_nonneg<T>(base: T, exp: u32) -> T
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

pub(crate) fn max(vec: &ArrayView1<f64>) -> f64 {
    vec.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

pub(crate) fn norm(vec: &ArrayView1<f64>) -> f64 {
    vec.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub(crate) fn normalize(vec: &ArrayView1<f64>) -> Array1<f64> {
    let norm = norm(vec);
    if norm == 0.0 {
        return vec.to_owned();
    }
    vec.map(|x| x / norm)
}

/// Compute the binomial coefficients up to `limit`.
pub(crate) fn binomial(limit: u64) -> Vec<Vec<u64>> {
    let mut result = vec![Vec::new(); limit as usize + 1];
    result[0] = vec![1];
    for n in 1..=limit as usize {
        result[n] = vec![1; n + 1];
        for k in 1..n {
            result[n][k] = result[n - 1][k - 1] + result[n - 1][k];
        }
    }
    result
}

/// Compute the Stirling numbers of the second kind up to `limit`.
pub(crate) fn stirling_second(limit: u64) -> Vec<Vec<u64>> {
    let mut result = vec![Vec::new(); limit as usize + 1];
    result[0] = vec![1];
    for n in 1..=limit as usize {
        result[n] = vec![0; n + 1];
        for k in 1..n {
            result[n][k] = k as u64 * result[n - 1][k] + result[n - 1][k - 1];
        }
        result[n][n] = 1;
    }
    result
}

/// Computes the values of the polylogarithm function Li_{-n}(x) for `n in 0..=limit`.
/// This is useful for computing the moments of the geometric distribution.
pub(crate) fn polylog_neg<T>(limit: u64, x: T) -> Vec<T>
where
    T: Zero
        + One
        + Clone
        + From<u64>
        + std::ops::AddAssign
        + std::ops::Sub<Output = T>
        + std::ops::MulAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let mut result = Vec::new();
    let stirling_second = stirling_second(limit + 1);
    let frac = x.clone() / (T::one() - x);
    for n in 0..=limit {
        let mut sum = T::zero();
        let mut k_factorial = T::one();
        let mut frac_power = T::one();
        for k in 0..=n {
            frac_power *= frac.clone();
            k_factorial *= T::from(k.max(1));
            sum += k_factorial.clone()
                * T::from(stirling_second[(n + 1) as usize][(k + 1) as usize])
                * frac_power.clone();
        }
        result.push(sum);
    }
    result
}

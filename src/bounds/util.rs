use std::ops::{Div, MulAssign};

use num_traits::One;

pub fn f64_to_mantissa_exponent(f: f64) -> (i64, i64) {
    if !f.is_finite() {
        unreachable!("Non-finite f64 in constraint: {f}");
    }
    let i = f as i64;
    if i as f64 == f {
        return (i, 0);
    }
    let bits: u64 = f.to_bits();
    let sign: i64 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = if exponent == 0 {
        (bits & 0x000f_ffff_ffff_ffff) << 1
    } else {
        (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
    } as i64;
    let trailing_zeros: i64 = mantissa.trailing_zeros().min(63).into();
    // Exponent bias + mantissa shift
    exponent -= 1023 + 52 - trailing_zeros;
    let mantissa = sign * (mantissa >> trailing_zeros);
    (mantissa, exponent)
}

pub fn f64_to_reduced_mantissa_exponent(f: f64) -> (i64, i64) {
    let (mantissa, exponent) = f64_to_mantissa_exponent(f);
    let mut mantissa = mantissa;
    let mut exponent = exponent;
    while mantissa % 2 == 0 && mantissa != 0 && exponent < 0 {
        mantissa /= 2;
        exponent += 1;
    }
    (mantissa, exponent)
}

pub fn f64_to_fraction(f: f64) -> (i128, u128) {
    let (mantissa, exponent) = f64_to_reduced_mantissa_exponent(f);
    let n = f as i128;
    if n as f64 == f {
        (n, 1)
    } else if (-127..=0).contains(&exponent) {
        (
            i128::from(mantissa),
            1u128 << u128::from((-exponent) as u64),
        )
    } else {
        todo!()
    }
}

pub fn f64_to_z3<'a>(ctx: &'a z3::Context, f: &f64) -> z3::ast::Real<'a> {
    let (mantissa, exponent) = f64_to_reduced_mantissa_exponent(*f);
    let m = z3::ast::Int::from_i64(ctx, mantissa).to_real();
    if exponent != 0 {
        let two = z3::ast::Int::from_i64(ctx, 2).to_real();
        let e = z3::ast::Int::from_i64(ctx, exponent).to_real();
        m * two.power(&e)
    } else {
        m
    }
}

pub fn z3_real_to_f64(real: &z3::ast::Real) -> Option<f64> {
    if let Some((n, d)) = real.as_real() {
        return Some(n as f64 / d as f64);
    }
    let string = real.to_string();
    if let Ok(f) = string.parse::<f64>() {
        Some(f)
    } else {
        let words = string.split_whitespace().collect::<Vec<_>>();
        if words.len() == 3 && words[0] == "(/" && words[2].ends_with(')') {
            let n = words[1].parse::<f64>().ok()?;
            let d = words[2][..words[2].len() - 1].parse::<f64>().ok()?;
            return Some(n / d);
        }
        None
    }
}

pub fn f64_to_qepcad(f: f64) -> String {
    let (mantissa, exponent) = f64_to_reduced_mantissa_exponent(f);
    let n = f as i128;
    if n as f64 == f {
        if n < 0 {
            format!("({n})")
        } else {
            n.to_string()
        }
    } else if (-127..=0).contains(&exponent) {
        format!("({mantissa}/{})", (1i128 << i128::from(-exponent)))
    } else {
        todo!()
    }
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

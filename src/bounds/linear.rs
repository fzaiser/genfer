use crate::bounds::sym_expr::SymExpr;

#[derive(Clone, Debug)]
pub struct LinearExpr {
    pub coeffs: Vec<f64>,
    pub constant: f64,
}

impl LinearExpr {
    pub fn new(coeffs: Vec<f64>, constant: f64) -> Self {
        Self { coeffs, constant }
    }

    pub fn zero() -> Self {
        Self::new(vec![], 0.0)
    }

    pub fn one() -> Self {
        Self::new(vec![1.0], 0.0)
    }

    pub fn constant(constant: f64) -> Self {
        Self::new(vec![], constant)
    }

    pub fn var(var: usize) -> Self {
        let mut coeffs = vec![0.0; var + 1];
        coeffs[var] = 1.0;
        Self::new(coeffs, 0.0)
    }

    pub fn as_constant(&self) -> Option<f64> {
        if self.coeffs.iter().all(|c| c == &0.0) {
            Some(self.constant)
        } else {
            None
        }
    }

    pub fn to_lp_expr(&self, vars: &[good_lp::Variable]) -> good_lp::Expression {
        let mut result = good_lp::Expression::from(self.constant);
        for (coeff, var) in self.coeffs.iter().zip(vars) {
            result.add_mul(*coeff, var);
        }
        result
    }
}

impl std::fmt::Display for LinearExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if *coeff == 0.0 {
                continue;
            }
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            if *coeff == 1.0 {
                write!(f, "{}", SymExpr::var(i))?;
            } else if *coeff == -1.0 {
                write!(f, "-{}", SymExpr::var(i))?;
            } else {
                write!(f, "{}{}", coeff, SymExpr::var(i))?;
            }
        }
        if self.constant != 0.0 {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}", self.constant)?;
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl std::ops::Neg for LinearExpr {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self * (-1.0)
    }
}

impl std::ops::Add for LinearExpr {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let constant = self.constant + other.constant;
        let (mut coeffs, other) = if self.coeffs.len() > other.coeffs.len() {
            (self.coeffs, other.coeffs)
        } else {
            (other.coeffs, self.coeffs)
        };
        for i in 0..other.len() {
            coeffs[i] += other[i];
        }
        Self::new(coeffs, constant)
    }
}

impl std::ops::Sub for LinearExpr {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl std::ops::Mul<f64> for LinearExpr {
    type Output = Self;

    #[inline]
    fn mul(self, other: f64) -> Self::Output {
        Self::new(
            self.coeffs.into_iter().map(|c| c * other).collect(),
            self.constant * other,
        )
    }
}

#[derive(Clone, Debug)]
pub struct LinearConstraint {
    expr: LinearExpr,
    /// If true, `expr` must be equal to zero, otherwise it must be non-positive
    eq_zero: bool,
}

impl LinearConstraint {
    pub fn eq(e1: LinearExpr, e2: LinearExpr) -> Self {
        Self {
            expr: e1 - e2,
            eq_zero: true,
        }
    }

    pub fn le(e1: LinearExpr, e2: LinearExpr) -> Self {
        Self {
            expr: e1 - e2,
            eq_zero: false,
        }
    }

    pub fn to_lp_constraint(&self, var_list: &[good_lp::Variable]) -> good_lp::Constraint {
        let result = self.expr.to_lp_expr(var_list);
        if self.eq_zero {
            result.eq(0.0)
        } else {
            result.leq(0.0)
        }
    }

    pub fn eval_constant(&self) -> Option<bool> {
        let constant = self.expr.as_constant()?;
        if self.eq_zero {
            Some(constant == 0.0)
        } else {
            Some(constant <= 0.0)
        }
    }
}

impl std::fmt::Display for LinearConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.eq_zero {
            write!(f, "{} = 0", self.expr)
        } else {
            write!(f, "{} <= 0", self.expr)
        }
    }
}

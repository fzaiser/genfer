use std::{
    ops::{AddAssign, SubAssign},
    time::Duration,
};

use good_lp::{variable, ProblemVariables, Solution, SolverModel};
use ndarray::{arr0, indices, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dimension, Slice};
use num_traits::{One, Zero};

use crate::{
    number::{Number, F64},
    ppl::{Distribution, Event, Natural, Program, Statement, Var},
};

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
}

pub struct BoundCtx {
    program_var_count: usize,
    sym_var_count: usize,
    // Variables used nonlinearly, in [0,1)
    nonlinear_param_vars: Vec<usize>,
    constraints: Vec<SymConstraint>,
    soft_constraints: Vec<SymConstraint>,
}

impl BoundCtx {
    pub fn new() -> Self {
        Self {
            program_var_count: 0,
            sym_var_count: 0,
            nonlinear_param_vars: Vec::new(),
            constraints: Vec::new(),
            soft_constraints: Vec::new(),
        }
    }

    pub fn sym_var_count(&self) -> usize {
        self.sym_var_count
    }

    pub fn constraints(&self) -> &[SymConstraint] {
        &self.constraints
    }

    pub fn soft_constraints(&self) -> &[SymConstraint] {
        &self.soft_constraints
    }

    pub fn fresh_sym_var_idx(&mut self) -> usize {
        let var = self.sym_var_count;
        self.sym_var_count += 1;
        var
    }

    pub fn fresh_sym_var(&mut self) -> SymExpr {
        SymExpr::var(self.fresh_sym_var_idx())
    }

    pub fn add_constraint(&mut self, constraint: SymConstraint) {
        // println!("Adding constraint {constraint}");
        self.constraints.push(constraint);
    }

    pub fn add_soft_constraint(&mut self, constraint: SymConstraint) {
        // println!("Adding soft constraint {constraint}");
        self.soft_constraints.push(constraint);
    }

    pub fn new_geo_param_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx();
        let var = SymExpr::var(idx);
        self.nonlinear_param_vars.push(idx);
        self.add_constraint(var.clone().must_ge(SymExpr::zero()));
        self.add_constraint(var.clone().must_le(SymExpr::one()));
        self.add_soft_constraint(var.clone().must_lt(SymExpr::one()));
        var
    }

    pub fn add_bounds(&mut self, lhs: GeometricBound, rhs: GeometricBound) -> GeometricBound {
        let count = self.program_var_count;
        let mut geo_params = Vec::with_capacity(count);
        for i in 0..count {
            let a = lhs.geo_params[i].clone();
            let b = rhs.geo_params[i].clone();
            if a == b {
                geo_params.push(a);
            } else {
                let new_geo_param_var = self.new_geo_param_var();
                geo_params.push(new_geo_param_var.clone());
                self.add_constraint(a.clone().must_le(new_geo_param_var.clone()));
                self.add_constraint(b.clone().must_le(new_geo_param_var.clone()));
                // The following constraint is not necessary, but it helps the solver:
                // We require the new variable to be the maximum of the other two.
                self.add_constraint(SymConstraint::or(vec![
                    new_geo_param_var.clone().must_eq(a),
                    new_geo_param_var.must_eq(b),
                ]));
            }
        }
        let polynomial = lhs.polynomial + rhs.polynomial; // TODO: check whether this is correct
        GeometricBound {
            polynomial,
            geo_params,
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    pub fn bound_event(
        &mut self,
        bound: GeometricBound,
        event: &Event,
    ) -> (GeometricBound, GeometricBound) {
        match event {
            Event::InSet(v, set) => {
                let alpha = bound.geo_params[v.id()].clone();
                let one_minus_alpha_v =
                    SymPolynomial::one() - SymPolynomial::var(*v) * alpha.clone();
                let mut then_bound = GeometricBound {
                    polynomial: SymPolynomial::zero(),
                    geo_params: bound.geo_params.clone(),
                };
                then_bound.geo_params[v.id()] = SymExpr::zero();
                let mut else_bound = bound;
                for Natural(n) in set {
                    let mut coeff = SymPolynomial::zero();
                    for i in 0..=*n {
                        coeff += else_bound.polynomial.coeff_of_var_power(*v, i as usize)
                            * alpha.clone().pow((n - i) as i32);
                    }
                    let monomial = coeff.clone() * SymPolynomial::var_power(*v, *n);
                    then_bound.polynomial += monomial.clone();
                    else_bound.polynomial -= monomial.clone() * one_minus_alpha_v.clone();
                }
                (then_bound, else_bound)
            }
            Event::DataFromDist(data, dist) => {
                if let Distribution::Bernoulli(p) = dist {
                    let p_compl = p.complement();
                    let p = F64::from_ratio(p.numer, p.denom).into();
                    let p_compl = F64::from_ratio(p_compl.numer, p_compl.denom).into();
                    let (then, els) = match data.0 {
                        0 => (p_compl, p),
                        1 => (p, p_compl),
                        _ => (0.0, 1.0),
                    };
                    let mut then_bound = bound.clone();
                    let mut else_bound = bound;
                    then_bound.polynomial *= SymExpr::from(then);
                    else_bound.polynomial *= SymExpr::from(els);
                    (then_bound, else_bound)
                } else {
                    todo!()
                }
            }
            Event::VarComparison(..) => todo!(),
            Event::Complement(event) => {
                let (then_bound, else_bound) = self.bound_event(bound, event);
                (else_bound, then_bound)
            }
            Event::Intersection(events) => {
                let mut then_bound = bound.clone();
                let mut else_bound = GeometricBound::zero(bound.geo_params.len());
                for event in events {
                    let (new_then, new_else) = self.bound_event(then_bound, event);
                    then_bound = new_then;
                    else_bound = self.add_bounds(else_bound, new_else);
                }
                (then_bound, else_bound)
            }
        }
    }

    pub fn bound_statement(&mut self, bound: GeometricBound, stmt: &Statement) -> GeometricBound {
        match stmt {
            Statement::Sample {
                var,
                distribution,
                add_previous_value,
            } => {
                let mut new_bound = if *add_previous_value {
                    bound
                } else {
                    bound.marginalize(*var)
                };
                match distribution {
                    Distribution::Bernoulli(p) => {
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        new_bound.polynomial *=
                            SymPolynomial::var(*var) * SymExpr::from(p) + (1.0 - p).into();
                        new_bound
                    }
                    Distribution::Geometric(p) if !add_previous_value => {
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        new_bound.polynomial *= SymExpr::from(p);
                        new_bound.geo_params[var.id()] = SymExpr::from(1.0 - p);
                        new_bound
                    }
                    Distribution::Uniform { start, end } => {
                        let mut factor = SymPolynomial::zero();
                        let len = f64::from(end.0 - start.0);
                        for i in start.0..end.0 {
                            factor += SymPolynomial::var_power(*var, i) / len.into();
                        }
                        new_bound.polynomial *= factor;
                        new_bound
                    }
                    _ => todo!(),
                }
            }
            Statement::Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                if let (None, Natural(n)) = (addend, offset) {
                    let mut new_bound = if *add_previous_value {
                        bound
                    } else {
                        bound.marginalize(*var)
                    };
                    new_bound.polynomial *= SymPolynomial::var_power(*var, *n);
                    new_bound
                } else {
                    todo!()
                }
            }
            Statement::Decrement { var, amount } => {
                let mut cur_bound = bound;
                let alpha = cur_bound.geo_params[var.id()].clone();
                for _ in 0..amount.0 {
                    let polynomial = cur_bound.polynomial;
                    let (p0, shifted) = polynomial.extract_zero_and_shift_left(*var);
                    let other = p0
                        * (SymPolynomial::one()
                            + (SymPolynomial::one() - SymPolynomial::var(*var)) * alpha.clone());
                    cur_bound.polynomial = shifted + other;
                }
                cur_bound
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_bound, else_bound) = self.bound_event(bound, cond);
                let then_bound = self.bound_statements(then_bound, then);
                let else_bound = self.bound_statements(else_bound, els);
                self.add_bounds(then_bound, else_bound)
            }
            Statement::Fail => GeometricBound::zero(self.program_var_count),
            Statement::Normalize { .. } => todo!(),
            Statement::While { cond, unroll, body } => {
                let mut bound = bound;
                let mut rest_bound = GeometricBound::zero(self.program_var_count);
                let unroll_count = unroll.unwrap_or(0);
                println!("Unrolling {unroll_count} times");
                for _ in 0..unroll_count {
                    let (then_bound, else_bound) = self.bound_event(bound.clone(), cond);
                    bound = self.bound_statements(then_bound, body);
                    rest_bound = self.add_bounds(rest_bound, else_bound);
                }
                let max_degree_p1 = 2; // TODO: should be configurable
                let (bound, else_bound) = self.bound_event(bound, cond);
                rest_bound = self.add_bounds(rest_bound, else_bound);
                let invariant = self.new_bound(max_degree_p1);
                self.assert_le(&bound, &invariant);
                let idx = self.fresh_sym_var_idx();
                let c = SymExpr::var(idx);
                self.nonlinear_param_vars.push(idx);
                self.add_constraint(c.clone().must_ge(SymExpr::zero()));
                self.add_constraint(c.clone().must_le(SymExpr::one()));
                self.add_soft_constraint(c.clone().must_lt(SymExpr::one()));
                let mut cur_bound = invariant.clone();
                for stmt in body {
                    cur_bound = self.bound_statement(cur_bound, stmt);
                }
                let (post_loop_bound, exit_bound) = self.bound_event(cur_bound, cond);
                self.assert_le(&post_loop_bound, &(invariant.clone() * c.clone()));
                let loop_bound = exit_bound / (SymExpr::one() - c);
                self.add_bounds(rest_bound, loop_bound)
            }
        }
    }

    pub fn bound_statements(
        &mut self,
        bound: GeometricBound,
        stmts: &[Statement],
    ) -> GeometricBound {
        let mut cur_bound = bound;
        for stmt in stmts {
            cur_bound = self.bound_statement(cur_bound, stmt);
        }
        cur_bound
    }

    pub fn bound_program(&mut self, program: &Program) -> GeometricBound {
        self.program_var_count = program.used_vars().num_vars();
        let init_bound = GeometricBound {
            polynomial: SymPolynomial::one(),
            geo_params: vec![SymExpr::zero(); self.program_var_count],
        };
        self.bound_statements(init_bound, &program.stmts)
    }

    pub fn output_smt(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "(set-logic QF_NRA)").unwrap();
        writeln!(out, "(set-option :pp.decimal true)").unwrap();
        writeln!(out).unwrap();
        for i in 0..self.sym_var_count() {
            writeln!(out, "(declare-const {} Real)", SymExpr::var(i)).unwrap();
        }
        writeln!(out).unwrap();
        for constraint in &self.constraints {
            writeln!(out, "(assert {})", constraint).unwrap();
        }
        for constraint in &self.soft_constraints {
            writeln!(out, "(assert-soft {})", constraint).unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "(check-sat)").unwrap();
        writeln!(out, "(get-model)").unwrap();
        out
    }

    fn new_polynomial(&mut self, degree_p1: usize) -> SymPolynomial {
        let shape = vec![degree_p1; self.program_var_count];
        let mut coeffs = ArrayD::zeros(shape);
        for c in &mut coeffs {
            *c = self.fresh_sym_var();
        }
        SymPolynomial::new(coeffs)
    }

    fn new_bound(&mut self, degree_p1: usize) -> GeometricBound {
        let mut geo_params = vec![SymExpr::zero(); self.program_var_count];
        for p in &mut geo_params {
            *p = self.new_geo_param_var();
        }
        GeometricBound {
            polynomial: self.new_polynomial(degree_p1),
            geo_params,
        }
    }

    fn assert_le_helper(
        &mut self,
        lhs_coeffs: &ArrayD<SymExpr>,
        lhs_geo_params: &[SymExpr],
        rhs_coeffs: &ArrayD<SymExpr>,
        rhs_geo_params: &[SymExpr],
    ) {
        if lhs_geo_params.is_empty() {
            assert!(lhs_coeffs.len() == 1);
            assert!(rhs_coeffs.len() == 1);
            let lhs_coeff = lhs_coeffs.first().unwrap().clone();
            let rhs_coeff = rhs_coeffs.first().unwrap().clone();
            self.add_constraint(lhs_coeff.must_le(rhs_coeff));
            return;
        }
        let len_lhs = if lhs_coeffs.ndim() == 0 {
            1
        } else {
            lhs_coeffs.len_of(Axis(0))
        };
        let len_rhs = if rhs_coeffs.ndim() == 0 {
            1
        } else {
            rhs_coeffs.len_of(Axis(0))
        };
        let len = len_lhs.max(len_rhs);
        let lhs_shape = lhs_coeffs.shape();
        let rhs_shape = rhs_coeffs.shape();
        let mut lhs_inequality = if lhs_coeffs.ndim() == 0 {
            ArrayD::zeros(&[][..])
        } else {
            ArrayD::zeros(&lhs_shape[1..])
        };
        let mut rhs_inequality = if rhs_coeffs.ndim() == 0 {
            ArrayD::zeros(&[][..])
        } else {
            ArrayD::zeros(&rhs_shape[1..])
        };
        for i in 0..len {
            lhs_inequality.mapv_inplace(|c| c * lhs_geo_params[0].clone());
            rhs_inequality.mapv_inplace(|c| c * rhs_geo_params[0].clone());
            if lhs_coeffs.ndim() == 0 {
                if i == 0 {
                    *lhs_inequality.first_mut().unwrap() += lhs_coeffs.first().unwrap().clone();
                }
            } else if i < len_lhs {
                lhs_inequality += &lhs_coeffs.index_axis(Axis(0), i);
            }
            if rhs_coeffs.ndim() == 0 {
                if i == 0 {
                    *rhs_inequality.first_mut().unwrap() += rhs_coeffs.first().unwrap().clone();
                }
            } else if i < len_rhs {
                rhs_inequality += &rhs_coeffs.index_axis(Axis(0), i);
            }
            self.assert_le_helper(
                &lhs_inequality,
                &lhs_geo_params[1..],
                &rhs_inequality,
                &rhs_geo_params[1..],
            );
        }
    }

    fn assert_le(&mut self, lhs: &GeometricBound, rhs: &GeometricBound) {
        // println!("Asserting less than:\n{lhs}\n<=\n{rhs}");
        for i in 0..self.program_var_count {
            self.add_constraint(lhs.geo_params[i].clone().must_le(rhs.geo_params[i].clone()));
        }
        self.assert_le_helper(
            &lhs.polynomial.coeffs,
            &lhs.geo_params,
            &rhs.polynomial.coeffs,
            &rhs.geo_params,
        );
    }

    fn solve_lp_for_nonlinear_var_assignment(
        &self,
        bound: &GeometricBound,
        nonlinear_var_assignment: &[f64],
    ) -> Result<GeometricBound, SolverError> {
        let mut replacements = (0..self.sym_var_count())
            .map(SymExpr::var)
            .collect::<Vec<_>>();
        for (i, v) in self.nonlinear_param_vars.iter().enumerate() {
            replacements[*v] = nonlinear_var_assignment[i].into();
        }
        let bound = bound.substitute(&replacements);
        let constraints = self
            .constraints
            .iter()
            .map(|c| c.substitute(&replacements))
            .collect::<Vec<_>>();
        let linear_constraints = constraints
            .iter()
            .map(|constraint| {
                constraint.extract_linear().unwrap_or_else(|| {
                    panic!("Constraint is not linear in the program variables: {constraint}")
                })
            })
            .collect::<Vec<_>>();
        let mut lp = ProblemVariables::new();
        let mut var_list = Vec::new();
        for replacement in replacements {
            match replacement {
                SymExpr::Variable(_) => {
                    var_list.push(lp.add_variable());
                }
                SymExpr::Constant(c) => {
                    var_list.push(lp.add(variable().min(c).max(c)));
                }
                _ => unreachable!(),
            }
        }
        let objective = bound
            .polynomial
            .coeffs
            .iter()
            .fold(good_lp::Expression::from(0.), |acc, c| {
                acc + c.extract_linear().unwrap().to_lp_expr(&var_list)
            });
        let mut problem = lp.minimise(objective).using(good_lp::default_solver);
        for constraint in &linear_constraints {
            problem.add_constraint(constraint.to_lp_constraint(&var_list));
        }
        let solution = problem.solve().map_err(|err| match err {
            good_lp::ResolutionError::Unbounded => panic!("Optimal solution is unbounded."),
            good_lp::ResolutionError::Infeasible => SolverError::Infeasible,
            good_lp::ResolutionError::Other(msg) => todo!("Other error: {msg}"),
            good_lp::ResolutionError::Str(msg) => todo!("Error: {msg}"),
        })?;
        let solution = var_list
            .iter()
            .map(|v| solution.value(*v))
            .collect::<Vec<_>>();
        let mut resolved_bound = bound;
        for coeff in &mut resolved_bound.polynomial.coeffs {
            let val = coeff.eval(&solution);
            *coeff = SymExpr::Constant(val);
        }
        for geo_param in &mut resolved_bound.geo_params {
            let val = geo_param.eval(&solution);
            *geo_param = SymExpr::Constant(val);
        }
        Ok(resolved_bound)
    }

    pub fn solve_z3(
        &self,
        bound: &GeometricBound,
        timeout: Duration,
    ) -> Result<GeometricBound, SolverError> {
        fn z3_real_to_f64(real: &z3::ast::Real) -> Option<f64> {
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
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(timeout.as_millis() as u64);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);
        for constraint in self.constraints() {
            solver.assert(&constraint.to_z3(&ctx))
        }
        for constraint in self.soft_constraints() {
            solver.assert(&constraint.to_z3(&ctx))
        }
        match solver.check() {
            z3::SatResult::Unknown => {
                if let Some(reason) = solver.get_reason_unknown() {
                    if reason == "timeout" {
                        return Err(SolverError::Timeout);
                    }
                    panic!("Solver responded 'unknown': {reason}")
                } else {
                    panic!("Solver responded 'unknown' but no reason was given.")
                }
            }
            z3::SatResult::Unsat => return Err(SolverError::Infeasible),
            z3::SatResult::Sat => {}
        }
        let model = solver
            .get_model()
            .unwrap_or_else(|| panic!("SMT solver's model is not available"));
        for var in 0..self.sym_var_count() {
            let val = model
                .eval(&z3::ast::Real::new_const(&ctx, var as u32), false)
                .unwrap();
            let val =
                z3_real_to_f64(&val).unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
            println!("{var} -> {val}", var = SymExpr::var(var));
        }
        let mut resolved_bound = bound.clone();
        for coeff in &mut resolved_bound.polynomial.coeffs {
            let val = model.eval(&coeff.to_z3(&ctx), false).unwrap();
            let val =
                z3_real_to_f64(&val).unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
            *coeff = SymExpr::Constant(val);
        }
        for geo_param in &mut resolved_bound.geo_params {
            let val = model.eval(&geo_param.to_z3(&ctx), false).unwrap();
            let val =
                z3_real_to_f64(&val).unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
            *geo_param = SymExpr::Constant(val);
        }
        println!("SMT solution:\n {resolved_bound}");
        println!("Optimizing polynomial coefficients...");
        let nonlinear_var_assignment = self
            .nonlinear_param_vars
            .iter()
            .map(|var| {
                let val = model.eval(&SymExpr::var(*var).to_z3(&ctx), false).unwrap();
                z3_real_to_f64(&val).unwrap_or_else(|| panic!("{val} cannot be converted to f64"))
            })
            .collect::<Vec<_>>();
        let optimized_solution =
            self.solve_lp_for_nonlinear_var_assignment(bound, &nonlinear_var_assignment)?;
        Ok(optimized_solution)
    }
}

impl Default for BoundCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometricBound {
    polynomial: SymPolynomial,
    geo_params: Vec<SymExpr>,
}

impl GeometricBound {
    fn zero(n: usize) -> Self {
        GeometricBound {
            polynomial: SymPolynomial::zero(),
            geo_params: vec![SymExpr::zero(); n],
        }
    }

    pub fn marginalize(&self, var: Var) -> Self {
        let mut polynomial = self.polynomial.clone();
        let mut geo_params = self.geo_params.clone();
        polynomial =
            polynomial.marginalize(var) * (SymExpr::one() - geo_params[var.id()].clone()).inverse();
        geo_params[var.id()] = SymExpr::zero();
        Self {
            polynomial,
            geo_params,
        }
    }

    fn substitute(&self, replacements: &[SymExpr]) -> GeometricBound {
        Self {
            polynomial: self.polynomial.substitute(replacements),
            geo_params: self
                .geo_params
                .iter()
                .map(|p| p.substitute(replacements))
                .collect(),
        }
    }
}

impl std::fmt::Display for GeometricBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.polynomial)?;
        writeln!(f, "______________________________________________________")?;
        for (i, param) in self.geo_params.iter().enumerate() {
            write!(f, "(1 - {param} * {})", Var(i))?;
        }
        writeln!(f)
    }
}

impl std::ops::Mul<SymExpr> for GeometricBound {
    type Output = Self;

    fn mul(mut self, rhs: SymExpr) -> Self::Output {
        self.polynomial *= rhs;
        self
    }
}

impl std::ops::Div<SymExpr> for GeometricBound {
    type Output = Self;

    fn div(mut self, rhs: SymExpr) -> Self::Output {
        self.polynomial /= rhs;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr {
    Constant(f64),
    Variable(usize),
    Add(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    Pow(Box<SymExpr>, i32),
}

impl SymExpr {
    pub fn var(i: usize) -> Self {
        Self::Variable(i)
    }

    pub fn inverse(self) -> Self {
        self.pow(-1)
    }

    pub fn pow(self, n: i32) -> Self {
        if n == 0 {
            Self::one()
        } else if n == 1 || (n >= 0 && self.is_zero()) || self.is_one() {
            self
        } else {
            Self::Pow(Box::new(self), n)
        }
    }

    /// Must equal `rhs`.
    pub fn must_eq(self, rhs: Self) -> SymConstraint {
        SymConstraint::Eq(self, rhs)
    }

    /// Must be less than `rhs`.
    pub fn must_lt(self, rhs: Self) -> SymConstraint {
        SymConstraint::Lt(self, rhs)
    }

    /// Must be less than or equal to `rhs`.
    pub fn must_le(self, rhs: Self) -> SymConstraint {
        SymConstraint::Le(self, rhs)
    }

    /// Must be greater than `rhs`.
    pub fn must_gt(self, rhs: Self) -> SymConstraint {
        SymConstraint::Lt(rhs, self)
    }

    /// Must be greater than or equal to `rhs`.
    pub fn must_ge(self, rhs: Self) -> SymConstraint {
        SymConstraint::Le(rhs, self)
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> Self {
        match self {
            SymExpr::Constant(_) => self.clone(),
            SymExpr::Variable(i) => replacements[*i].clone(),
            SymExpr::Add(lhs, rhs) => lhs.substitute(replacements) + rhs.substitute(replacements),
            SymExpr::Mul(lhs, rhs) => lhs.substitute(replacements) * rhs.substitute(replacements),
            SymExpr::Pow(base, n) => base.substitute(replacements).pow(*n),
        }
    }

    pub fn extract_linear(&self) -> Option<LinearExpr> {
        match self {
            SymExpr::Constant(c) => Some(LinearExpr::constant(*c)),
            SymExpr::Variable(i) => Some(LinearExpr::var(*i)),
            SymExpr::Add(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                Some(lhs + rhs)
            }
            SymExpr::Mul(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                if let Some(factor) = lhs.as_constant() {
                    Some(rhs * factor)
                } else if let Some(factor) = rhs.as_constant() {
                    Some(lhs * factor)
                } else {
                    None
                }
            }
            SymExpr::Pow(base, n) => {
                if *n == 0 {
                    return Some(LinearExpr::constant(1.0));
                }
                let base = base.extract_linear()?;
                if let Some(base) = base.as_constant() {
                    return Some(LinearExpr::constant(base.powi(*n)));
                }
                if *n == 1 {
                    Some(base)
                } else {
                    None
                }
            }
        }
    }

    pub fn to_z3<'a>(&self, ctx: &'a z3::Context) -> z3::ast::Real<'a> {
        match self {
            SymExpr::Constant(f) => {
                if !f.is_finite() {
                    unreachable!("Non-finite f64 in constraint: {f}");
                }
                let bits: u64 = f.to_bits();
                let sign: i64 = if bits >> 63 == 0 { 1 } else { -1 };
                let mut exponent = ((bits >> 52) & 0x7ff) as i64;
                let mantissa = if exponent == 0 {
                    (bits & 0x000f_ffff_ffff_ffff) << 1
                } else {
                    (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
                } as i64;
                // Exponent bias + mantissa shift
                exponent -= 1023 + 52;
                let m = z3::ast::Int::from_i64(ctx, sign * mantissa).to_real();
                let two = z3::ast::Int::from_i64(ctx, 2).to_real();
                let e = z3::ast::Int::from_i64(ctx, exponent).to_real();
                m * two.power(&e)
            }
            SymExpr::Variable(v) => z3::ast::Real::new_const(ctx, *v as u32),
            SymExpr::Add(e1, e2) => e1.to_z3(ctx) + e2.to_z3(ctx),
            SymExpr::Mul(e1, e2) => e1.to_z3(ctx) * e2.to_z3(ctx),
            SymExpr::Pow(e, n) => e
                .to_z3(ctx)
                .power(&z3::ast::Int::from_i64(ctx, (*n).into()).to_real()),
        }
    }

    fn eval(&self, values: &[f64]) -> f64 {
        match self {
            SymExpr::Constant(c) => *c,
            SymExpr::Variable(v) => values[*v],
            SymExpr::Add(lhs, rhs) => lhs.eval(values) + rhs.eval(values),
            SymExpr::Mul(lhs, rhs) => lhs.eval(values) * rhs.eval(values),
            SymExpr::Pow(base, n) => base.eval(values).powi(*n),
        }
    }
}

impl From<f64> for SymExpr {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}

impl Zero for SymExpr {
    fn zero() -> Self {
        Self::Constant(0.0)
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_zero(),
            _ => false,
        }
    }
}

impl One for SymExpr {
    fn one() -> Self {
        Self::Constant(1.0)
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_one(),
            _ => false,
        }
    }
}

impl std::ops::Neg for SymExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else if let Self::Constant(c) = self {
            (-c).into()
        } else {
            self * (-1.0).into()
        }
    }
}

impl std::ops::Add for SymExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            Self::Add(Box::new(self), Box::new(rhs))
        }
    }
}

impl std::ops::AddAssign for SymExpr {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Sub for SymExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self == rhs {
            Self::zero()
        } else {
            self + (-rhs)
        }
    }
}

impl std::ops::SubAssign for SymExpr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl std::ops::Mul for SymExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else if self.is_one() {
            rhs
        } else if rhs.is_one() {
            self
        } else {
            Self::Mul(Box::new(self), Box::new(rhs))
        }
    }
}

impl std::ops::Div for SymExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl std::fmt::Display for SymExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(value) => {
                if value < &0.0 {
                    write!(f, "(- {})", -value)
                } else {
                    write!(f, "{value}")
                }
            }
            Self::Variable(i) => write!(f, "x{i}"),
            Self::Add(lhs, rhs) => write!(f, "(+ {lhs} {rhs})"),
            Self::Mul(lhs, rhs) => {
                if Self::Constant(-1.0) == **rhs {
                    write!(f, "(- {lhs})")
                } else {
                    write!(f, "(* {lhs} {rhs})")
                }
            }
            Self::Pow(expr, n) => {
                if *n == -1 {
                    write!(f, "(/ 1 {expr})")
                } else if *n < 0 {
                    write!(f, "(/ 1 (^ {expr} {}))", -n)
                } else {
                    write!(f, "(^ {expr} {n})")
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum SymConstraint {
    Eq(SymExpr, SymExpr),
    Lt(SymExpr, SymExpr),
    Le(SymExpr, SymExpr),
    Or(Vec<SymConstraint>),
}
impl SymConstraint {
    pub fn or(constraints: Vec<SymConstraint>) -> Self {
        Self::Or(constraints)
    }

    fn to_z3<'a>(&self, ctx: &'a z3::Context) -> z3::ast::Bool<'a> {
        match self {
            SymConstraint::Eq(e1, e2) => z3::ast::Ast::_eq(&e1.to_z3(ctx), &e2.to_z3(ctx)),
            SymConstraint::Lt(e1, e2) => e1.to_z3(ctx).lt(&e2.to_z3(ctx)),
            SymConstraint::Le(e1, e2) => e1.to_z3(ctx).le(&e2.to_z3(ctx)),
            SymConstraint::Or(constraints) => {
                let disjuncts = constraints.iter().map(|c| c.to_z3(ctx)).collect::<Vec<_>>();
                z3::ast::Bool::or(ctx, &disjuncts.iter().collect::<Vec<_>>())
            }
        }
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> SymConstraint {
        match self {
            SymConstraint::Eq(e1, e2) => {
                SymConstraint::Eq(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Lt(e1, e2) => {
                SymConstraint::Lt(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Le(e1, e2) => {
                SymConstraint::Le(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Or(constraints) => SymConstraint::Or(
                constraints
                    .iter()
                    .map(|c| c.substitute(replacements))
                    .collect(),
            ),
        }
    }

    fn extract_linear(&self) -> Option<LinearConstraint> {
        match self {
            SymConstraint::Eq(e1, e2) => Some(LinearConstraint::eq(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Lt(..) => None,
            SymConstraint::Le(e1, e2) => Some(LinearConstraint::le(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Or(constraints) => {
                // Here we only support constraints without variables
                for constraint in constraints {
                    if let Some(linear_constraint) = constraint.extract_linear() {
                        if linear_constraint.eval_constant() == Some(true) {
                            return Some(LinearConstraint::eq(
                                LinearExpr::constant(0.0),
                                LinearExpr::constant(0.0),
                            ));
                        }
                    }
                }
                return None;
            }
        }
    }
}

impl std::fmt::Display for SymConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq(e1, e2) => write!(f, "(= {e1} {e2})"),
            Self::Lt(e1, e2) => write!(f, "(< {e1} {e2})"),
            Self::Le(e1, e2) => write!(f, "(<= {e1} {e2})"),
            Self::Or(constraints) => {
                write!(f, "(or")?;
                for constraint in constraints {
                    write!(f, " {}", constraint)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SymPolynomial {
    coeffs: ArrayD<SymExpr>,
}

impl SymPolynomial {
    #[inline]
    pub fn new(coeffs: ArrayD<SymExpr>) -> Self {
        Self { coeffs }
    }

    pub fn var(var: Var) -> Self {
        Self::var_power(var, 1)
    }

    pub fn var_power(Var(v): Var, n: u32) -> SymPolynomial {
        if n == 0 {
            return Self::one();
        }
        let mut shape = vec![1; v + 1];
        shape[v] = n as usize + 1;
        let mut coeffs = ArrayD::zeros(shape);
        *coeffs
            .index_axis_mut(Axis(v), n as usize)
            .first_mut()
            .unwrap() = SymExpr::one();
        Self { coeffs }
    }

    #[inline]
    fn max_shape(&self, other: &Self) -> Vec<usize> {
        let mut shape = vec![1; self.coeffs.ndim().max(other.coeffs.ndim())];
        for (v, dim) in shape.iter_mut().enumerate() {
            if v < self.coeffs.ndim() {
                *dim = (*dim).max(self.coeffs.len_of(Axis(v)));
            }
            if v < other.coeffs.ndim() {
                *dim = (*dim).max(other.coeffs.len_of(Axis(v)));
            }
        }
        shape
    }

    #[inline]
    fn sum_shape(&self, other: &Self) -> Vec<usize> {
        let mut shape = vec![1; self.coeffs.ndim().max(other.coeffs.ndim())];
        for (v, dim) in shape.iter_mut().enumerate() {
            if v < self.coeffs.ndim() {
                *dim += self.coeffs.len_of(Axis(v)) - 1;
            }
            if v < other.coeffs.ndim() {
                *dim += other.coeffs.len_of(Axis(v)) - 1;
            }
        }
        shape
    }

    fn marginalize(&self, Var(v): Var) -> SymPolynomial {
        let mut result_shape = self.coeffs.shape().to_vec();
        if v >= result_shape.len() {
            return self.clone();
        }
        result_shape[v] = 1;
        let mut result = ArrayD::zeros(result_shape);
        for coeff in self.coeffs.axis_chunks_iter(Axis(v), 1) {
            result += &coeff;
        }
        Self::new(result)
    }

    fn coeff_of_var_power(&self, Var(v): Var, order: usize) -> SymPolynomial {
        if v >= self.coeffs.ndim() {
            if order == 0 {
                return self.clone();
            }
            return Self::zero();
        }
        if order >= self.coeffs.len_of(Axis(v)) {
            return Self::zero();
        }
        Self::new(
            self.coeffs
                .slice_axis(Axis(v), Slice::from(order..=order))
                .to_owned(),
        )
    }

    fn shift_coeffs_left(&self, Var(v): Var) -> SymPolynomial {
        Self::new(self.coeffs.slice_axis(Axis(v), Slice::from(1..)).to_owned())
    }

    fn extract_zero_and_shift_left(self, var: Var) -> (SymPolynomial, SymPolynomial) {
        let p0 = self.coeff_of_var_power(var, 0);
        let rest = self - p0.clone();
        let shifted = rest.shift_coeffs_left(var);
        (p0, shifted)
    }

    fn substitute(&self, replacements: &[SymExpr]) -> SymPolynomial {
        Self::new(self.coeffs.map(|c| c.substitute(replacements)))
    }
}

impl From<f64> for SymPolynomial {
    #[inline]
    fn from(value: f64) -> Self {
        Self::new(arr0(value.into()).into_dyn())
    }
}

impl Zero for SymPolynomial {
    #[inline]
    fn zero() -> Self {
        Self::new(arr0(SymExpr::zero()).into_dyn())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(SymExpr::is_zero)
    }
}

impl One for SymPolynomial {
    #[inline]
    fn one() -> Self {
        Self::new(arr0(SymExpr::one()).into_dyn())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs.first().unwrap().is_one()
    }
}

impl std::ops::Mul<SymExpr> for SymPolynomial {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: SymExpr) -> Self::Output {
        Self::new(self.coeffs.mapv(|c| c * rhs.clone()))
    }
}

impl std::ops::MulAssign<SymExpr> for SymPolynomial {
    #[inline]
    fn mul_assign(&mut self, rhs: SymExpr) {
        self.coeffs.mapv_inplace(|c| c * rhs.clone());
    }
}

impl std::ops::Div<SymExpr> for SymPolynomial {
    type Output = Self;

    #[inline]
    fn div(self, rhs: SymExpr) -> Self::Output {
        Self::new(self.coeffs.mapv(|c| c / rhs.clone()))
    }
}

impl std::ops::DivAssign<SymExpr> for SymPolynomial {
    #[inline]
    fn div_assign(&mut self, rhs: SymExpr) {
        self.coeffs.mapv_inplace(|c| c / rhs.clone());
    }
}

impl std::ops::Neg for SymPolynomial {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.coeffs.mapv_inplace(|c| -c);
        self
    }
}

#[inline]
fn broadcast<T>(xs: &mut ArrayD<T>, ys: &mut ArrayD<T>) {
    if xs.ndim() < ys.ndim() {
        for i in xs.ndim()..ys.ndim() {
            xs.insert_axis_inplace(Axis(i));
        }
    }
    if ys.ndim() < xs.ndim() {
        for i in ys.ndim()..xs.ndim() {
            ys.insert_axis_inplace(Axis(i));
        }
    }
}

impl std::ops::Add for SymPolynomial {
    type Output = Self;

    #[inline]
    fn add(mut self, mut other: Self) -> Self::Output {
        let result_shape = self.max_shape(&other);
        let mut result = ArrayD::zeros(result_shape);
        broadcast(&mut self.coeffs, &mut other.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .add_assign(&other.coeffs);
        Self::new(result)
    }
}

impl std::ops::AddAssign for SymPolynomial {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl std::ops::Sub for SymPolynomial {
    type Output = Self;

    #[inline]
    fn sub(mut self, mut other: Self) -> Self::Output {
        let result_shape = self.max_shape(&other);
        let mut result = ArrayD::zeros(result_shape);
        broadcast(&mut self.coeffs, &mut other.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .sub_assign(&other.coeffs);
        Self::new(result)
    }
}

impl std::ops::SubAssign for SymPolynomial {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl std::ops::Mul for SymPolynomial {
    type Output = Self;

    #[inline]
    fn mul(mut self, mut other: Self) -> Self::Output {
        fn mul_helper(
            xs: &ArrayViewD<SymExpr>,
            ys: &ArrayViewD<SymExpr>,
            res: &mut ArrayViewMutD<SymExpr>,
        ) {
            if res.is_empty() {
                return;
            }
            if res.ndim() == 0 {
                *res.first_mut().unwrap() +=
                    xs.first().unwrap().clone() * ys.first().unwrap().clone();
                return;
            }
            for (k, mut z) in res.axis_iter_mut(Axis(0)).enumerate() {
                let lo = (k + 1).saturating_sub(ys.len_of(Axis(0)));
                let hi = (k + 1).min(xs.len_of(Axis(0)));
                for j in lo..hi {
                    mul_helper(
                        &xs.index_axis(Axis(0), j),
                        &ys.index_axis(Axis(0), k - j),
                        &mut z,
                    );
                }
            }
        }

        // Recognize multiplication by zero:
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        // Broadcast to common shape:
        broadcast(&mut self.coeffs, &mut other.coeffs);
        let result_shape = self.sum_shape(&other);

        // Recognize multiplication by one:
        if self.is_one() {
            return other;
        }
        if other.is_one() {
            return self;
        }

        let mut result = ArrayD::zeros(result_shape);
        mul_helper(
            &self.coeffs.view(),
            &other.coeffs.view(),
            &mut result.view_mut(),
        );
        Self::new(result)
    }
}

impl std::ops::MulAssign for SymPolynomial {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl std::fmt::Display for SymPolynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.coeffs.shape();
        let mut first = true;
        let mut empty_output = true;
        for index in indices(shape) {
            if self.coeffs[&index].is_zero() {
                continue;
            }
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}", self.coeffs[&index])?;
            empty_output = false;
            for (i, exponent) in index.as_array_view().into_iter().enumerate() {
                if *exponent == 0 {
                    continue;
                }
                write!(f, "{}", crate::ppl::Var(i))?;
                if *exponent > 1 {
                    write!(f, "^{exponent}")?;
                }
            }
        }
        if empty_output {
            write!(f, "0")?;
        }
        Ok(())
    }
}

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
struct LinearConstraint {
    expr: LinearExpr,
    /// If true, `expr` must be equal to zero, otherwise it must be non-positive
    eq_zero: bool,
}

impl LinearConstraint {
    fn eq(e1: LinearExpr, e2: LinearExpr) -> Self {
        Self {
            expr: e1 - e2,
            eq_zero: true,
        }
    }

    fn le(e1: LinearExpr, e2: LinearExpr) -> Self {
        Self {
            expr: e1 - e2,
            eq_zero: false,
        }
    }

    fn to_lp_constraint(&self, var_list: &[good_lp::Variable]) -> good_lp::Constraint {
        let result = self.expr.to_lp_expr(var_list);
        if self.eq_zero {
            result.eq(0.0)
        } else {
            result.leq(0.0)
        }
    }

    fn eval_constant(&self) -> Option<bool> {
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

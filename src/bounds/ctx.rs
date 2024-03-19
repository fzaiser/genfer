use std::time::Duration;

use good_lp::{variable, ProblemVariables, Solution, SolverModel};
use ndarray::{ArrayD, Axis};
use num_traits::{One, Zero};

use crate::{
    bounds::{
        bound::{BoundResult, GeometricBound},
        sym_expr::{SymConstraint, SymExpr},
        sym_poly::SymPolynomial,
    },
    number::{Number, F64},
    ppl::{Distribution, Event, Natural, Program, Statement},
    semantics::{
        support::{SupportTransformer, VarSupport},
        Transformer,
    },
    support::SupportSet,
};

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
}

pub struct BoundCtx {
    support: SupportTransformer,
    program_var_count: usize,
    sym_var_count: usize,
    // Variables used nonlinearly, in [0,1)
    nonlinear_param_vars: Vec<usize>,
    constraints: Vec<SymConstraint>,
    soft_constraints: Vec<SymConstraint>,
}

impl Default for BoundCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for BoundCtx {
    type Domain = BoundResult;

    fn init(&mut self, program: &Program) -> Self::Domain {
        self.program_var_count = program.used_vars().num_vars();
        let init_bound = GeometricBound {
            polynomial: SymPolynomial::one(),
            geo_params: vec![SymExpr::zero(); self.program_var_count],
        };
        BoundResult {
            bound: init_bound,
            var_supports: self.support.init(program),
        }
    }

    fn transform_event(
        &mut self,
        event: &Event,
        init: Self::Domain,
    ) -> (Self::Domain, Self::Domain) {
        match event {
            Event::InSet(v, set) => {
                let alpha = init.bound.geo_params[v.id()].clone();
                let one_minus_alpha_v =
                    SymPolynomial::one() - SymPolynomial::var(*v) * alpha.clone();
                let mut then_bound = GeometricBound {
                    polynomial: SymPolynomial::zero(),
                    geo_params: init.bound.geo_params.clone(),
                };
                then_bound.geo_params[v.id()] = SymExpr::zero();
                let mut then_res = BoundResult {
                    bound: then_bound,
                    var_supports: init.var_supports.clone(),
                };
                let mut else_res = init;
                (then_res.var_supports, else_res.var_supports) =
                    self.support.transform_event(event, else_res.var_supports);
                for Natural(n) in set {
                    let mut coeff = SymPolynomial::zero();
                    for i in 0..=*n {
                        coeff += else_res.bound.polynomial.coeff_of_var_power(*v, i as usize)
                            * alpha.clone().pow((n - i) as i32);
                    }
                    let monomial = coeff.clone() * SymPolynomial::var_power(*v, *n);
                    then_res.bound.polynomial += monomial.clone();
                    else_res.bound.polynomial -= monomial.clone() * one_minus_alpha_v.clone();
                }
                (then_res, else_res)
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
                    let mut then_res = init.clone();
                    let mut else_res = init;
                    then_res.bound.polynomial *= SymExpr::from(then);
                    else_res.bound.polynomial *= SymExpr::from(els);
                    (then_res, else_res)
                } else {
                    todo!()
                }
            }
            Event::VarComparison(..) => todo!(),
            Event::Complement(event) => {
                let (then_res, else_res) = self.transform_event(event, init);
                (else_res, then_res)
            }
            Event::Intersection(events) => {
                let mut else_res = BoundResult {
                    bound: GeometricBound::zero(init.bound.geo_params.len()),
                    var_supports: VarSupport::Empty(init.var_supports.num_vars()),
                };
                let mut then_res = init;
                for event in events {
                    let (new_then, new_else) = self.transform_event(event, then_res);
                    then_res = new_then;
                    else_res = self.add_bound_results(else_res, new_else);
                }
                (then_res, else_res)
            }
        }
    }

    fn transform_statement(&mut self, stmt: &Statement, init: Self::Domain) -> Self::Domain {
        let direct_var_supports = if cfg!(debug_assertions) {
            Some(
                self.support
                    .transform_statement(stmt, init.var_supports.clone()),
            )
        } else {
            None
        };
        let result = match stmt {
            Statement::Sample {
                var,
                distribution,
                add_previous_value,
            } => {
                let new_var_info = SupportTransformer::transform_distribution(
                    distribution,
                    *var,
                    init.var_supports.clone(),
                    *add_previous_value,
                );
                let mut res = if *add_previous_value {
                    init
                } else {
                    init.marginalize(*var)
                };
                match distribution {
                    Distribution::Bernoulli(p) => {
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        res.bound.polynomial *=
                            SymPolynomial::var(*var) * SymExpr::from(p) + (1.0 - p).into();
                    }
                    Distribution::Geometric(p) if !add_previous_value => {
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        res.bound.polynomial *= SymExpr::from(p);
                        res.bound.geo_params[var.id()] = SymExpr::from(1.0 - p);
                    }
                    Distribution::Uniform { start, end } => {
                        let mut factor = SymPolynomial::zero();
                        let len = f64::from(end.0 - start.0);
                        for i in start.0..end.0 {
                            factor += SymPolynomial::var_power(*var, i) / len.into();
                        }
                        res.bound.polynomial *= factor;
                    }
                    _ => todo!(),
                };
                res.var_supports = new_var_info;
                res
            }
            Statement::Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                if let (None, Natural(n)) = (addend, offset) {
                    let mut new_bound = if *add_previous_value {
                        init
                    } else {
                        init.marginalize(*var)
                    };
                    new_bound.bound.polynomial *= SymPolynomial::var_power(*var, *n);
                    new_bound.var_supports = self
                        .support
                        .transform_statement(stmt, new_bound.var_supports);
                    new_bound
                } else {
                    todo!()
                }
            }
            Statement::Decrement { var, offset } => {
                let mut cur = init;
                let alpha = cur.bound.geo_params[var.id()].clone();
                for _ in 0..offset.0 {
                    let polynomial = cur.bound.polynomial;
                    let (p0, shifted) = polynomial.extract_zero_and_shift_left(*var);
                    let other = p0
                        * (SymPolynomial::one()
                            + (SymPolynomial::one() - SymPolynomial::var(*var)) * alpha.clone());
                    cur.bound.polynomial = shifted + other;
                }
                cur.var_supports = self.support.transform_statement(stmt, cur.var_supports);
                cur
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_res, else_res) = self.transform_event(cond, init);
                let then_res = self.transform_statements(then, then_res);
                let else_res = self.transform_statements(els, else_res);
                self.add_bound_results(then_res, else_res)
            }
            Statement::While { cond, unroll, body } => self.bound_while(cond, *unroll, body, init),
            Statement::Fail => BoundResult {
                bound: GeometricBound::zero(self.program_var_count),
                var_supports: VarSupport::empty(init.var_supports.num_vars()),
            },
            Statement::Normalize { .. } => todo!(),
        };
        if let Some(direct_var_info) = direct_var_supports {
            debug_assert_eq!(
                result.var_supports, direct_var_info,
                "inconsistent variable support info for:\n{stmt}"
            );
        }
        result
    }
}

impl BoundCtx {
    pub fn new() -> Self {
        Self {
            support: SupportTransformer,
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

    pub fn add_bound_results(&mut self, lhs: BoundResult, rhs: BoundResult) -> BoundResult {
        let bound = self.add_bounds(lhs.bound, rhs.bound);
        let var_supports = lhs.var_supports.join(&rhs.var_supports);
        BoundResult {
            bound,
            var_supports,
        }
    }

    fn bound_while(
        &mut self,
        cond: &Event,
        unroll: Option<usize>,
        body: &[Statement],
        init: BoundResult,
    ) -> BoundResult {
        let mut pre_loop = init;
        let mut rest = BoundResult {
            bound: GeometricBound::zero(self.program_var_count),
            var_supports: VarSupport::empty(pre_loop.var_supports.num_vars()),
        };
        let unroll_count = unroll.unwrap_or(0);
        let (iters, invariant_supports, _) =
            self.support
                .analyze_while(cond, body, pre_loop.var_supports.clone());
        let unroll_count = if let Some(iters) = iters {
            unroll_count.max(iters)
        } else {
            unroll_count
        };
        println!("Unrolling {unroll_count} times");
        for _ in 0..unroll_count {
            let (then_bound, else_bound) = self.transform_event(cond, pre_loop.clone());
            pre_loop = self.transform_statements(body, then_bound);
            rest = self.add_bound_results(rest, else_bound);
        }
        let min_degree = 1;
        let shape = match &invariant_supports {
            VarSupport::Empty(num_vars) => vec![0; *num_vars],
            VarSupport::Prod(supports) => supports
                .iter()
                .map(|s| match s {
                    SupportSet::Empty => 0,
                    SupportSet::Range { end, .. } => {
                        if let Some(end) = end {
                            *end as usize + 1
                        } else {
                            min_degree
                        }
                    }
                    _ => todo!(),
                })
                .collect::<Vec<_>>(),
        };
        let finite_supports = match &invariant_supports {
            VarSupport::Empty(num_vars) => vec![true; *num_vars],
            VarSupport::Prod(supports) => supports
                .iter()
                .map(|s| s.is_empty() || s.finite_nonempty_range().is_some())
                .collect::<Vec<_>>(),
        };
        let (_, invariant_supports, _) =
            self.support
                .analyze_while(cond, body, pre_loop.var_supports.clone());
        let (loop_entry, loop_exit) = self.transform_event(cond, pre_loop);
        rest = self.add_bound_results(rest, loop_exit);
        let invariant = BoundResult {
            bound: self.new_bound(shape, &finite_supports),
            var_supports: invariant_supports,
        };
        println!("Invariant: {invariant}");
        self.assert_le(&loop_entry.bound, &invariant.bound);
        let idx = self.fresh_sym_var_idx();
        let c = SymExpr::var(idx);
        println!("Invariant-c: {c}");
        self.nonlinear_param_vars.push(idx);
        self.add_constraint(c.clone().must_ge(SymExpr::zero()));
        self.add_constraint(c.clone().must_le(SymExpr::one()));
        self.add_soft_constraint(c.clone().must_lt(SymExpr::one()));
        let cur_bound = self.transform_statements(&body, invariant.clone());
        let (post_loop, mut exit) = self.transform_event(cond, cur_bound);
        self.assert_le(&post_loop.bound, &(invariant.bound.clone() * c.clone()));
        exit.bound = exit.bound / (SymExpr::one() - c.clone());
        self.add_bound_results(exit, rest)
    }

    pub fn output_python(&self, bound: &GeometricBound) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "import numpy as np").unwrap();
        writeln!(out, "from scipy.optimize import *").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "nvars = {}", self.sym_var_count).unwrap();
        writeln!(out, "bounds = Bounds(").unwrap();
        write!(out, "  [").unwrap();
        for i in 0..self.sym_var_count {
            if self.nonlinear_param_vars.contains(&i) {
                write!(out, "0, ").unwrap();
            } else {
                write!(out, "-np.inf, ").unwrap();
            }
        }
        writeln!(out, "],").unwrap();
        write!(out, "  [").unwrap();
        for i in 0..self.sym_var_count {
            if self.nonlinear_param_vars.contains(&i) {
                write!(out, "1, ").unwrap();
            } else {
                write!(out, "np.inf, ").unwrap();
            }
        }
        writeln!(out, "])").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "def constraint_fun(x):").unwrap();
        writeln!(out, "  return [").unwrap();
        for c in &self.constraints {
            match c {
                SymConstraint::Eq(_, _) | SymConstraint::Or(_) => continue,
                SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                    writeln!(out, "    ({}) - ({}),", rhs.to_python(), lhs.to_python()).unwrap();
                }
            }
        }
        writeln!(out, "  ]").unwrap();
        writeln!(out, "constraints = NonlinearConstraint(constraint_fun, 0, np.inf, jac='2-point', hess=BFGS())").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "def objective(x):").unwrap();
        let numer = bound
            .polynomial
            .coeffs
            .iter()
            .fold(SymExpr::zero(), |acc, c| acc + c.clone());
        write!(out, "  return ({}) / (", numer.to_python()).unwrap();
        for e in &bound.geo_params {
            write!(out, "(1 - {}) * ", e.to_python()).unwrap();
        }
        writeln!(out, "1)").unwrap();
        writeln!(out, "x0 = np.full((nvars,), 0.9)").unwrap();
        out
    }

    pub fn output_python_z3(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "import z3").unwrap();
        writeln!(out, "z3.set_option(precision=5)").unwrap();
        writeln!(out).unwrap();
        for i in 0..self.sym_var_count {
            writeln!(out, "x{i} = Real('x{i}')").unwrap();
        }
        writeln!(out, "s = Solver()").unwrap();
        for constraint in &self.constraints {
            writeln!(out, "s.add({})", constraint.to_python_z3()).unwrap();
        }
        for constraint in &self.soft_constraints {
            writeln!(out, "s.add({})", constraint.to_python_z3()).unwrap();
        }
        writeln!(out).unwrap();
        out
    }

    pub fn output_smt<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        writeln!(out, "(set-logic QF_NRA)")?;
        writeln!(out, "(set-option :pp.decimal true)")?;
        writeln!(out)?;
        for i in 0..self.sym_var_count() {
            writeln!(out, "(declare-const {} Real)", SymExpr::var(i))?;
        }
        writeln!(out)?;
        for constraint in &self.constraints {
            writeln!(out, "(assert {constraint})")?;
        }
        for constraint in &self.soft_constraints {
            writeln!(out, "(assert-soft {constraint})")?;
        }
        writeln!(out)?;
        writeln!(out, "(check-sat)")?;
        writeln!(out, "(get-model)")?;
        Ok(())
    }

    pub fn output_qepcad<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        // Name:
        writeln!(out, "[Geometric bounds constraints]")?;

        // List of variables:
        write!(out, "(")?;
        let mut first = true;
        for i in 0..self.sym_var_count {
            if first {
                first = false;
            } else {
                write!(out, ", ")?;
            }
            write!(out, "{}", crate::ppl::Var(i))?;
        }
        writeln!(out, ")")?;

        // Number of free variables:
        writeln!(out, "2")?; // two variables for plotting

        // Formula:
        for i in 2..self.sym_var_count {
            writeln!(out, "(E {})", crate::ppl::Var(i))?;
        }
        writeln!(out, "[")?;
        let mut first = true;
        for c in self.constraints() {
            if first {
                first = false;
            } else {
                writeln!(out, r" /\")?;
            }
            write!(out, "  {}", c.to_qepcad())?;
        }
        for c in self.soft_constraints() {
            writeln!(out, r" /\")?;
            write!(out, "  {}", c.to_qepcad())?;
        }
        writeln!(out, "\n].")?;

        // Commands for various solving stages:
        writeln!(out, "go")?;
        writeln!(out, "go")?;
        writeln!(out, "go")?;
        writeln!(out, "p-2d-cad 0 1 0 1 0.0001 plot.eps")?; // 2D plot
        writeln!(out, "go")?;
        Ok(())
    }

    fn new_polynomial(&mut self, shape: Vec<usize>) -> SymPolynomial {
        let mut coeffs = ArrayD::zeros(shape);
        for c in &mut coeffs {
            *c = self.fresh_sym_var();
        }
        SymPolynomial::new(coeffs)
    }

    fn new_bound(&mut self, shape: Vec<usize>, finite_supports: &[bool]) -> GeometricBound {
        let mut geo_params = vec![SymExpr::zero(); shape.len()];
        for (v, p) in geo_params.iter_mut().enumerate() {
            if !finite_supports[v] {
                *p = self.new_geo_param_var();
            }
        }
        GeometricBound {
            polynomial: self.new_polynomial(shape),
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
        optimize: bool,
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
            solver.assert(&constraint.to_z3(&ctx));
        }
        for constraint in self.soft_constraints() {
            solver.assert(&constraint.to_z3(&ctx));
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
        let objective = bound.total_mass();
        let mut obj_lo = 0.0;
        let mut obj_hi;
        let model = loop {
            let model = solver
                .get_model()
                .unwrap_or_else(|| panic!("SMT solver's model is not available"));
            for var in 0..self.sym_var_count() {
                let val = model
                    .eval(&z3::ast::Real::new_const(&ctx, var as u32), false)
                    .unwrap();
                let val = z3_real_to_f64(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                println!("{var} -> {val}", var = SymExpr::var(var));
            }
            let mut resolved_bound = bound.clone();
            for coeff in &mut resolved_bound.polynomial.coeffs {
                let val = model.eval(&coeff.to_z3(&ctx), false).unwrap();
                let val = z3_real_to_f64(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                *coeff = SymExpr::Constant(val);
            }
            for geo_param in &mut resolved_bound.geo_params {
                let val = model.eval(&geo_param.to_z3(&ctx), false).unwrap();
                let val = z3_real_to_f64(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                *geo_param = SymExpr::Constant(val);
            }
            println!("SMT solution:\n {resolved_bound}");
            let obj_val =
                z3_real_to_f64(&model.eval(&objective.to_z3(&ctx), false).unwrap()).unwrap();
            println!("Total mass (objective): {obj_val}");
            obj_hi = obj_val;
            println!("Objective bound: [{obj_lo}, {obj_hi}]");
            if !optimize || obj_hi - obj_lo < 0.1 * obj_hi {
                break model;
            }
            solver.push();
            let mid = (obj_lo + obj_hi) / 2.0;
            solver.assert(&objective.to_z3(&ctx).le(&SymExpr::from(mid).to_z3(&ctx)));
            loop {
                match solver.check() {
                    z3::SatResult::Sat => {
                        break;
                    }
                    z3::SatResult::Unknown | z3::SatResult::Unsat => {
                        solver.pop(1);
                        obj_lo = mid;
                    }
                }
            }
        };
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

use ndarray::{ArrayD, ArrayViewD, Axis, Dimension, Slice};
use num_traits::{One, Zero};

use std::ops::AddAssign;

use crate::{
    bound::{Egd, FiniteDiscrete, GeometricBound},
    numbers::{Number, Rational},
    ppl::{Distribution, Event, Natural, Program, Statement, Var},
    semantics::{
        support::{SupportTransformer, VarSupport},
        Transformer,
    },
    support::SupportSet,
    sym_expr::{SymConstraint, SymExpr, SymExprKind},
};

pub struct GeometricBoundSemantics {
    verbose: bool,
    unroll: usize,
    min_degree: usize,
    evt: bool,
    do_while_transform: bool,
    support: SupportTransformer,
    program_var_count: usize,
    // Variables used nonlinearly, in [0,1)
    nonlinear_vars: Vec<usize>,
    decay_vars: Vec<usize>,
    factor_vars: Vec<usize>,
    block_vars: Vec<usize>,
    sym_var_bounds: Vec<(Rational, Rational)>,
    constraints: Vec<SymConstraint>,
}

impl Default for GeometricBoundSemantics {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for GeometricBoundSemantics {
    type Domain = GeometricBound;

    fn init(&mut self, program: &Program) -> Self::Domain {
        self.program_var_count = program.used_vars().num_vars();
        let lower = FiniteDiscrete {
            masses: ArrayD::ones(vec![1; self.program_var_count]),
        };
        let upper = Egd {
            block: ArrayD::ones(vec![1; self.program_var_count]),
            decays: vec![SymExpr::zero(); self.program_var_count],
        };
        let var_supports = self.support.init(program);
        GeometricBound {
            lower,
            upper,
            var_supports,
        }
    }

    fn transform_event(
        &mut self,
        event: &Event,
        mut init: Self::Domain,
    ) -> (Self::Domain, Self::Domain) {
        match event {
            Event::InSet(v, set) => {
                let max = set.iter().fold(0, |acc, x| acc.max(x.0 as usize));
                init.upper.extend_axis(*v, max + 2);
                let axis = Axis(v.id());
                let len_lower = init.lower.masses.len_of(axis);
                let len_upper = init.upper.block.len_of(axis);
                let mut then_lower_bound = init.lower.clone();
                let mut else_lower_bound = init.lower;
                let mut then_upper_bound = init.upper.clone();
                let mut else_upper_bound = init.upper;
                then_upper_bound.decays[v.id()] = SymExpr::zero();
                for i in 0..len_lower {
                    if set.contains(&Natural(i as u64)) {
                        else_lower_bound
                            .masses
                            .index_axis_mut(axis, i)
                            .fill(Rational::zero());
                    } else {
                        then_lower_bound
                            .masses
                            .index_axis_mut(axis, i)
                            .fill(Rational::zero());
                    }
                }
                for i in 0..len_upper {
                    if set.contains(&Natural(i as u64)) {
                        else_upper_bound
                            .block
                            .index_axis_mut(axis, i)
                            .fill(SymExpr::zero());
                    } else {
                        then_upper_bound
                            .block
                            .index_axis_mut(axis, i)
                            .fill(SymExpr::zero());
                    }
                }
                let (then_support, else_support) =
                    self.support.transform_event(event, init.var_supports);
                let then_res = GeometricBound {
                    lower: then_lower_bound,
                    upper: then_upper_bound,
                    var_supports: then_support,
                };
                let else_res = GeometricBound {
                    lower: else_lower_bound,
                    upper: else_upper_bound,
                    var_supports: else_support,
                };
                (then_res, else_res)
            }
            Event::DataFromDist(data, dist) => {
                if let Distribution::Bernoulli(p) = dist {
                    let p = Rational::from_ratio(p.numer, p.denom);
                    let p_compl = Rational::one() - p.clone();
                    let (then, els) = match data.0 {
                        0 => (p_compl, p),
                        1 => (p, p_compl),
                        _ => (Rational::zero(), Rational::one()),
                    };
                    let mut then_res = init.clone();
                    let mut else_res = init;
                    then_res.lower *= then.clone();
                    else_res.lower *= els.clone();
                    then_res.upper *= SymExpr::from(then);
                    else_res.upper *= SymExpr::from(els);
                    (then_res, else_res)
                } else {
                    todo!("Unsupported event: {event}")
                }
            }
            Event::VarComparison(..) => todo!("Comparison of two variables is not supported"),
            Event::Complement(event) => {
                let (then_res, else_res) = self.transform_event(event, init);
                (else_res, then_res)
            }
            Event::Intersection(events) => {
                let mut else_res = GeometricBound::zero(self.program_var_count);
                let mut then_res = init;
                for event in events {
                    let (new_then, new_else) = self.transform_event(event, then_res);
                    then_res = new_then;
                    else_res = self.add_bounds(else_res, new_else);
                }
                (then_res, else_res)
            }
        }
    }

    #[expect(clippy::too_many_lines)]
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
                    init.marginalize_out(*var)
                };
                match distribution {
                    Distribution::Bernoulli(p) => {
                        let p = Rational::from_ratio(p.numer, p.denom);
                        res.lower
                            .add_categorical(*var, &[Rational::one() - p.clone(), p.clone()]);
                        res.upper
                            .add_categorical(*var, &[Rational::one() - p.clone(), p]);
                    }
                    Distribution::Geometric(p) if !add_previous_value => {
                        let p = Rational::from_ratio(p.numer, p.denom);
                        let mut added_masses = res.lower.marginalize_out(*var) * p.clone();
                        res.lower = FiniteDiscrete::zero(res.var_supports.num_vars());
                        let limit = self.unroll + 1;
                        // TODO: should this limit be higher?
                        for _ in 0..limit {
                            res.lower += &added_masses;
                            added_masses *= Rational::one() - p.clone();
                            added_masses.shift_right(*var, 1);
                        }
                        res.upper *= SymExpr::from(p.clone());
                        res.upper.decays[var.id()] = SymExpr::from(Rational::one() - p);
                    }
                    Distribution::Uniform { start, end } => {
                        let p = Rational::one() / Rational::from_int(end.0 - start.0);
                        let categorical = (0..end.0)
                            .map(|x| {
                                if x < start.0 {
                                    Rational::zero()
                                } else {
                                    p.clone()
                                }
                            })
                            .collect::<Vec<_>>();
                        res.lower.add_categorical(*var, &categorical);
                        res.upper.add_categorical(*var, &categorical);
                    }
                    Distribution::Categorical(categorical) => {
                        let categorical = categorical
                            .iter()
                            .map(|p| Rational::from_ratio(p.numer, p.denom))
                            .collect::<Vec<_>>();
                        res.lower.add_categorical(*var, &categorical);
                        res.upper.add_categorical(*var, &categorical);
                    }
                    _ => todo!("Unsupported distribution: {distribution}"),
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
                let mut new_bound = if *add_previous_value {
                    init
                } else {
                    init.marginalize_out(*var)
                };
                match (addend, offset) {
                    (Some((Natural(1), w)), Natural(0)) => {
                        assert_ne!(var, w, "Cannot assign/add a variable to itself");
                        if let Some(range) = new_bound.var_supports[w].finite_nonempty_range() {
                            let mut new_lower_shape = new_bound.lower.masses.shape().to_vec();
                            let lower_len = new_lower_shape[var.id()];
                            let mut new_upper_shape = new_bound.upper.block.shape().to_vec();
                            new_lower_shape[var.id()] += *range.end() as usize;
                            new_upper_shape[var.id()] += *range.end() as usize;
                            let new_upper_len = new_upper_shape[var.id()];
                            new_bound.upper.extend_axis(*var, new_upper_len);
                            let lower_masses = &new_bound.lower.masses;
                            let upper_block = &new_bound.upper.block;
                            let mut res_lower_masses = ArrayD::zeros(new_lower_shape);
                            let mut res_upper_block = ArrayD::zeros(new_upper_shape);
                            for i in range {
                                let i = i as usize;
                                if i < res_lower_masses.len_of(Axis(w.id())) {
                                    res_lower_masses
                                        .slice_axis_mut(Axis(w.id()), Slice::from(i..=i))
                                        .slice_axis_mut(
                                            Axis(var.id()),
                                            Slice::from(i..lower_len + i),
                                        )
                                        .add_assign(
                                            &lower_masses
                                                .slice_axis(Axis(w.id()), Slice::from(i..=i))
                                                .slice_axis(Axis(var.id()), Slice::from(0..)),
                                        );
                                }
                                res_upper_block
                                    .slice_axis_mut(Axis(w.id()), Slice::from(i..=i))
                                    .slice_axis_mut(Axis(var.id()), Slice::from(i..new_upper_len))
                                    .add_assign(
                                        &upper_block
                                            .slice_axis(Axis(w.id()), Slice::from(i..=i))
                                            .slice_axis(
                                                Axis(var.id()),
                                                Slice::from(0..new_upper_len - i),
                                            ),
                                    );
                            }
                            new_bound.lower.masses = res_lower_masses;
                            new_bound.upper.block = res_upper_block;
                        } else {
                            todo!("Addition of a variable is not implemented for infinite support: {stmt}");
                        }
                    }
                    (None, offset) => {
                        new_bound.lower.shift_right(*var, offset.0 as usize);
                        new_bound.upper.shift_right(*var, offset.0 as usize);
                    }
                    _ => todo!("Unsupported assignment: {stmt}"),
                }
                new_bound.var_supports = self
                    .support
                    .transform_statement(stmt, new_bound.var_supports);
                new_bound
            }
            Statement::Decrement { var, offset } => {
                let mut new_bound = init;
                new_bound.lower.shift_left(*var, offset.0 as usize);
                new_bound.upper.shift_left(*var, offset.0 as usize);
                new_bound.var_supports = self
                    .support
                    .transform_statement(stmt, new_bound.var_supports);
                new_bound
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_res, else_res) = self.transform_event(cond, init);
                let then_res = self.transform_statements(then, then_res);
                let else_res = self.transform_statements(els, else_res);
                self.add_bounds(then_res, else_res)
            }
            Statement::While { cond, unroll, body } => self.bound_while(cond, *unroll, body, init),
            Statement::Fail => GeometricBound::zero(self.program_var_count),
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

impl GeometricBoundSemantics {
    pub fn new() -> Self {
        Self {
            verbose: false,
            unroll: 0,
            min_degree: 1,
            evt: false,
            do_while_transform: false,
            support: SupportTransformer::default(),
            program_var_count: 0,
            nonlinear_vars: Vec::new(),
            decay_vars: Vec::new(),
            factor_vars: Vec::new(),
            block_vars: Vec::new(),
            sym_var_bounds: Vec::new(),
            constraints: Vec::new(),
        }
    }

    pub fn with_min_degree(self, min_degree: usize) -> Self {
        Self { min_degree, ..self }
    }

    pub fn with_unroll(self, unroll: usize) -> Self {
        Self {
            unroll,
            support: self.support.with_unroll(unroll),
            ..self
        }
    }

    pub fn with_evt(self, evt: bool) -> Self {
        Self { evt, ..self }
    }

    pub fn with_do_while_transform(self, do_while_transform: bool) -> Self {
        Self {
            do_while_transform,
            ..self
        }
    }

    pub fn with_verbose(self, verbose: bool) -> Self {
        Self { verbose, ..self }
    }

    pub fn sym_var_count(&self) -> usize {
        self.sym_var_bounds.len()
    }

    pub fn nonlinear_vars(&self) -> &[usize] {
        &self.nonlinear_vars
    }

    pub fn geom_vars(&self) -> &[usize] {
        &self.decay_vars
    }

    pub fn factor_vars(&self) -> &[usize] {
        &self.factor_vars
    }

    pub fn block_vars(&self) -> &[usize] {
        &self.block_vars
    }

    pub fn sym_var_bounds(&self) -> &[(Rational, Rational)] {
        &self.sym_var_bounds
    }

    pub fn constraints(&self) -> &[SymConstraint] {
        &self.constraints
    }

    fn fresh_sym_var_idx(&mut self, lo: Rational, hi: Rational) -> usize {
        let var = self.sym_var_count();
        self.sym_var_bounds.push((lo, hi));
        var
    }

    pub(crate) fn add_constraint(&mut self, constraint: SymConstraint) {
        // Recognize variable bounds (lo <= v) constraints
        if let (SymExprKind::Constant(lo), SymExprKind::Variable(v)) =
            (constraint.lhs.kind(), constraint.rhs.kind())
        {
            let bound = &mut self.sym_var_bounds[*v];
            bound.0 = bound.0.max(&lo.rat());
            return;
        }
        if constraint.is_trivial() {
            return;
        }
        self.constraints.push(constraint);
    }

    pub(crate) fn new_decay_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx(Rational::zero(), Rational::one());
        let var = SymExpr::var(idx);
        self.nonlinear_vars.push(idx);
        self.decay_vars.push(idx);
        var
    }

    pub(crate) fn new_contraction_factor_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx(Rational::zero(), Rational::one());
        let var = SymExpr::var(idx);
        self.nonlinear_vars.push(idx);
        self.factor_vars.push(idx);
        var
    }

    pub(crate) fn new_block_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx(Rational::zero(), Rational::infinity());
        let var = SymExpr::var(idx);
        self.block_vars.push(idx);
        var
    }

    pub(crate) fn add_egds(&mut self, mut lhs: Egd, mut rhs: Egd) -> Egd {
        let count = self.program_var_count;
        let mut decays = Vec::with_capacity(count);
        for i in 0..count {
            let a = lhs.decays[i].clone();
            let b = rhs.decays[i].clone();
            if a == b {
                decays.push(a);
            } else if a.is_zero() {
                decays.push(b);
            } else if b.is_zero() {
                decays.push(a);
            } else {
                let new_decay_var = self.new_decay_var();
                decays.push(new_decay_var.clone());
                self.add_constraint(a.clone().must_le(new_decay_var.clone()));
                self.add_constraint(b.clone().must_le(new_decay_var.clone()));
            }
        }
        for i in 0..count {
            let lhs_len = lhs.block.len_of(Axis(i));
            let rhs_len = rhs.block.len_of(Axis(i));
            if lhs_len < rhs_len {
                lhs.extend_axis(Var(i), rhs_len);
            } else if rhs_len < lhs_len {
                rhs.extend_axis(Var(i), lhs_len);
            }
        }
        Egd {
            block: lhs.block + rhs.block,
            decays,
        }
    }

    pub(crate) fn add_bounds(
        &mut self,
        lhs: GeometricBound,
        rhs: GeometricBound,
    ) -> GeometricBound {
        let lower = &lhs.lower + &rhs.lower;
        let upper = self.add_egds(lhs.upper, rhs.upper);
        let var_supports = lhs.var_supports.join(&rhs.var_supports);
        GeometricBound {
            lower,
            upper,
            var_supports,
        }
    }

    fn bound_while(
        &mut self,
        cond: &Event,
        unroll: Option<usize>,
        body: &[Statement],
        init: GeometricBound,
    ) -> GeometricBound {
        let mut pre_loop = init;
        let mut rest = GeometricBound::zero(self.program_var_count);
        let unroll_count = unroll.unwrap_or(self.unroll);
        let unroll_result =
            self.support
                .find_unroll_fixpoint(cond, body, pre_loop.var_supports.clone());
        let unroll_count = if let Some((iters, _, _)) = unroll_result {
            unroll_count.max(iters)
        } else {
            unroll_count
        };
        if self.verbose {
            println!("\nUnrolling {unroll_count} times");
        }
        for _ in 0..unroll_count {
            let (then_bound, else_bound) = self.transform_event(cond, pre_loop.clone());
            pre_loop = self.transform_statements(body, then_bound);
            rest = self.add_bounds(rest, else_bound);
        }
        let invariant_supports =
            self.support
                .find_while_invariant(cond, body, pre_loop.var_supports.clone());
        let invariant_supports = if self.do_while_transform {
            let (invariant_entry, _) = self.support.transform_event(cond, invariant_supports);
            invariant_entry
        } else {
            invariant_supports
        };
        let shape = match &invariant_supports {
            VarSupport::Empty(num_vars) => vec![(0, Some(0)); *num_vars],
            VarSupport::Prod(supports) => supports
                .iter()
                .map(|s| match s {
                    SupportSet::Empty => (0, Some(0)),
                    SupportSet::Range { start, end } => {
                        (*start as usize, end.map(|x| x as usize + 1))
                    }
                })
                .collect::<Vec<_>>(),
        };
        if self.evt {
            let invariant = GeometricBound {
                lower: FiniteDiscrete::zero(self.program_var_count),
                upper: self.new_bound(shape, self.min_degree),
                var_supports: invariant_supports,
            };
            if self.verbose {
                println!("\nEVT-invariant:\n{invariant}");
            }
            let (loop_entry, loop_exit) = self.transform_event(cond, invariant.clone());
            let one_iter = self.transform_statements(body, loop_entry);
            let rhs = self.add_bounds(pre_loop, one_iter);
            if self.verbose {
                println!("\nPost loop body:\n{rhs}");
            }
            self.assert_le(&rhs.upper, &invariant.upper);
            self.add_bounds(loop_exit, rest)
        } else {
            let invariant = GeometricBound {
                lower: FiniteDiscrete::zero(self.program_var_count),
                upper: self.new_bound(shape, self.min_degree),
                var_supports: invariant_supports,
            };
            if self.verbose {
                println!("Invariant:\n{invariant}");
            }
            let (post_loop, mut exit) = if self.do_while_transform {
                let (loop_entry, loop_exit) = self.transform_event(cond, pre_loop);
                rest = self.add_bounds(rest, loop_exit);
                self.assert_le(&loop_entry.upper, &invariant.upper);
                let post_loop = self.transform_statements(body, invariant.clone());
                self.transform_event(cond, post_loop)
            } else {
                self.assert_le(&pre_loop.upper, &invariant.upper);
                let (loop_entry, loop_exit) = self.transform_event(cond, invariant.clone());
                let post_loop = self.transform_statements(body, loop_entry);
                (post_loop, loop_exit)
            };
            if self.verbose {
                println!("\nPost loop body:\n{post_loop}");
            }
            let c = self.new_contraction_factor_var();
            if self.verbose {
                println!("Invariant-c: {c}");
            }
            self.assert_le(&post_loop.upper, &(invariant.upper.clone() * c.clone()));
            exit.upper /= SymExpr::one() - c.clone();
            let result = self.add_bounds(exit, rest);
            if self.verbose {
                println!("\nResulting loop bound:\n{result}");
            }
            result
        }
    }

    fn new_block(&mut self, shape: Vec<(usize, usize)>) -> ArrayD<SymExpr> {
        let dims = shape.iter().map(|(_, e)| (*e).max(1)).collect::<Vec<_>>();
        let mut coeffs = ArrayD::zeros(dims);
        for (idx, c) in coeffs.indexed_iter_mut() {
            let is_nonzero = idx
                .as_array_view()
                .iter()
                .zip(&shape)
                .all(|(i, (start, end))| i >= start || (start >= end && i + 1 == *end));
            if is_nonzero {
                *c = self.new_block_var();
            }
        }
        coeffs
    }

    fn new_bound(&mut self, shape: Vec<(usize, Option<usize>)>, min_degree: usize) -> Egd {
        let mut decays = vec![SymExpr::zero(); shape.len()];
        for (v, p) in decays.iter_mut().enumerate() {
            if shape[v].1.is_none() {
                *p = self.new_decay_var();
            }
        }
        let shape = shape
            .into_iter()
            .map(|(start, end)| (start, end.unwrap_or(min_degree)))
            .collect::<Vec<_>>();
        Egd {
            block: self.new_block(shape),
            decays,
        }
    }

    fn assert_le_helper(
        &mut self,
        lhs_coeffs: &ArrayViewD<SymExpr>,
        lhs_factor: SymExpr,
        lhs_decays: &[SymExpr],
        rhs_coeffs: &ArrayViewD<SymExpr>,
        rhs_factor: SymExpr,
        rhs_decays: &[SymExpr],
    ) {
        if lhs_decays.is_empty() {
            assert!(lhs_coeffs.len() == 1);
            assert!(rhs_coeffs.len() == 1);
            let lhs_coeff = lhs_coeffs.first().unwrap().clone() * lhs_factor;
            let rhs_coeff = rhs_coeffs.first().unwrap().clone() * rhs_factor;
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
        for i in 0..len {
            let (lhs, lhs_factor) = if i < len_lhs {
                (lhs_coeffs.index_axis(Axis(0), i), lhs_factor.clone())
            } else {
                (
                    lhs_coeffs.index_axis(Axis(0), len_lhs - 1),
                    lhs_factor.clone() * lhs_decays[0].clone().pow((i - len_lhs + 1) as i32),
                )
            };
            let (rhs, rhs_factor) = if i < len_rhs {
                (rhs_coeffs.index_axis(Axis(0), i), rhs_factor.clone())
            } else {
                (
                    rhs_coeffs.index_axis(Axis(0), len_rhs - 1),
                    rhs_factor.clone() * rhs_decays[0].clone().pow((i - len_rhs + 1) as i32),
                )
            };
            self.assert_le_helper(
                &lhs,
                lhs_factor,
                &lhs_decays[1..],
                &rhs,
                rhs_factor,
                &rhs_decays[1..],
            );
        }
    }

    fn assert_le(&mut self, lhs: &Egd, rhs: &Egd) {
        // println!("Asserting less than:\n{lhs}\n<=\n{rhs}");
        for i in 0..self.program_var_count {
            self.add_constraint(lhs.decays[i].clone().must_le(rhs.decays[i].clone()));
        }
        self.assert_le_helper(
            &lhs.block.view(),
            SymExpr::one(),
            &lhs.decays,
            &rhs.block.view(),
            SymExpr::one(),
            &rhs.decays,
        );
    }
}

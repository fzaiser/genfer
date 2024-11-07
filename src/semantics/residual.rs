use ndarray::{ArrayD, Axis, Slice};
use num_traits::{One, Zero};
use std::ops::AddAssign;

use crate::{
    bound::{FiniteDiscrete, ResidualBound},
    numbers::{Number, Rational},
    ppl::{Distribution, Event, Natural, Program, Statement},
    semantics::{support::SupportTransformer, Transformer},
};

#[derive(Default)]
pub struct ResidualSemantics {
    unroll: usize,
    verbose: bool,
    support: SupportTransformer,
}

impl ResidualSemantics {
    pub fn with_unroll(mut self, unroll: usize) -> Self {
        self.unroll = unroll;
        self.support = self.support.with_unroll(unroll);
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Transformer for ResidualSemantics {
    type Domain = ResidualBound;

    fn init(&mut self, program: &Program) -> Self::Domain {
        let dist = FiniteDiscrete {
            masses: ArrayD::ones(vec![1; program.used_vars().num_vars()]),
        };
        let reject = Rational::zero();
        let var_supports = self.support.init(program);
        ResidualBound {
            lower: dist,
            reject,
            var_supports,
        }
    }

    fn transform_event(
        &mut self,
        event: &Event,
        init: Self::Domain,
    ) -> (Self::Domain, Self::Domain) {
        match event {
            Event::InSet(v, set) => {
                let axis = Axis(v.id());
                let len = init.lower.masses.len_of(axis);
                let mut then_lower = init.lower.clone();
                let mut else_lower = init.lower;
                for i in 0..len {
                    if set.contains(&Natural(i as u64)) {
                        else_lower
                            .masses
                            .index_axis_mut(axis, i)
                            .fill(Rational::zero());
                    } else {
                        then_lower
                            .masses
                            .index_axis_mut(axis, i)
                            .fill(Rational::zero());
                    }
                }
                let (then_support, else_support) =
                    self.support.transform_event(event, init.var_supports);
                let then_res = ResidualBound {
                    lower: then_lower,
                    reject: Rational::zero(),
                    var_supports: then_support,
                };
                let else_res = ResidualBound {
                    lower: else_lower,
                    reject: Rational::zero(),
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
                    then_res.reject = Rational::zero();
                    else_res.reject = Rational::zero();
                    (then_res, else_res)
                } else {
                    todo!("Unsupported event: {event}")
                }
            }
            Event::VarComparison(..) => todo!("Unsupported event: {event}"),
            Event::Complement(event) => {
                let (then_res, else_res) = self.transform_event(event, init);
                (else_res, then_res)
            }
            Event::Intersection(events) => {
                let mut else_res = ResidualBound::zero(init.var_count());
                let mut then_res = init;
                for event in events {
                    let (new_then, new_else) = self.transform_event(event, then_res);
                    then_res = new_then;
                    else_res += new_else;
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
                    }
                    Distribution::Geometric(p) if !add_previous_value => {
                        let var_count = res.var_count();
                        let p = Rational::from_ratio(p.numer, p.denom);
                        let mut added_masses = res.lower * p.clone();
                        res.lower = FiniteDiscrete::zero(var_count);
                        // TODO: this could be more efficient by pre-allocating the array
                        let limit = self.unroll + 1;
                        for _ in 0..limit {
                            res.lower += &added_masses;
                            added_masses *= Rational::one() - p.clone();
                            added_masses.shift_right(*var, 1);
                        }
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
                    }
                    Distribution::Categorical(categorical) => {
                        let categorical = categorical
                            .iter()
                            .map(|p| Rational::from_ratio(p.numer, p.denom))
                            .collect::<Vec<_>>();
                        res.lower.add_categorical(*var, &categorical);
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
                let mut res = if *add_previous_value {
                    init
                } else {
                    init.marginalize_out(*var)
                };
                match (addend, offset) {
                    (Some((Natural(1), w)), Natural(0)) => {
                        assert_ne!(var, w, "Cannot assign/add a variable to itself");
                        if let Some(range) = res.var_supports[w].finite_nonempty_range() {
                            let mut new_shape = res.lower.masses.shape().to_vec();
                            let len = new_shape[var.id()];
                            new_shape[var.id()] += *range.end() as usize;
                            let masses = &res.lower.masses;
                            let mut res_masses = ArrayD::zeros(new_shape);
                            for i in range {
                                let i = i as usize;
                                if i < res_masses.len_of(Axis(w.id())) {
                                    res_masses
                                        .slice_axis_mut(Axis(w.id()), Slice::from(i..=i))
                                        .slice_axis_mut(Axis(var.id()), Slice::from(i..len + i))
                                        .add_assign(
                                            &masses
                                                .slice_axis(Axis(w.id()), Slice::from(i..=i))
                                                .slice_axis(Axis(var.id()), Slice::from(0..)),
                                        );
                                }
                            }
                            res.lower.masses = res_masses;
                        } else {
                            todo!("Addition of a variable is not implemented for infinite support: {}", stmt.to_string());
                        }
                    }
                    (None, offset) => {
                        res.lower.shift_right(*var, offset.0 as usize);
                    }
                    _ => todo!("Unsupported assignment: {}", stmt.to_string()),
                }
                res.var_supports = self.support.transform_statement(stmt, res.var_supports);
                res
            }
            Statement::Decrement { var, offset } => {
                let mut new_bound = init;
                new_bound.lower.shift_left(*var, offset.0 as usize);
                new_bound.var_supports = self
                    .support
                    .transform_statement(stmt, new_bound.var_supports);
                new_bound
            }
            Statement::IfThenElse { cond, then, els } => {
                let reject = init.reject.clone();
                let (then_res, else_res) = self.transform_event(cond, init);
                let then_res = self.transform_statements(then, then_res);
                let else_res = self.transform_statements(els, else_res);
                (then_res + else_res).add_reject(reject)
            }
            Statement::While { cond, unroll, body } => {
                let mut pre_loop = init;
                let mut result = ResidualBound::zero(pre_loop.var_count());
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
                    println!("Unrolling {unroll_count} times");
                }
                for _ in 0..unroll_count {
                    let reject = pre_loop.reject.clone();
                    pre_loop.reject = Rational::zero();
                    let (then_bound, else_bound) = self.transform_event(cond, pre_loop.clone());
                    pre_loop = self.transform_statements(body, then_bound);
                    result += else_bound.add_reject(reject);
                }
                let invariant =
                    self.support
                        .find_while_invariant(cond, body, pre_loop.var_supports.clone());
                let (_, loop_exit) = self.support.transform_event(cond, invariant.clone());
                result.var_supports = result.var_supports.join(&loop_exit);
                result
            }
            Statement::Fail => {
                let mut result = ResidualBound::zero(init.var_count());
                result.reject = init.lower.total_mass();
                result
            }
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

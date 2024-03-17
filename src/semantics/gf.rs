use std::marker::PhantomData;

use crate::ppl::{Comparison, Distribution, Event, Natural, PosRatio, Program, Statement, Var};
use crate::{generating_function::GenFun, number::Number, support::SupportSet};

use super::{
    support::{SupportTransformer, VarSupport},
    Transformer,
};

const DEFAULT_UNROLL: usize = 8;

#[derive(Clone, Debug)]
pub struct GfTranslation<T> {
    pub var_info: VarSupport,
    pub gf: GenFun<T>,
    /// Remaining probability mass not captured in `gf`
    pub rest: GenFun<T>,
}

impl<T: Number> GfTranslation<T> {
    fn zero(num_vars: usize) -> Self {
        GfTranslation {
            var_info: VarSupport::empty(num_vars),
            gf: GenFun::zero(),
            rest: GenFun::zero(),
        }
    }

    /// Joins two translations, like two branches of an if-statement.
    ///
    /// There is a subtle difference to the `+` operation on translations.
    /// `join` takes the maximum of the remaining probability masses,
    /// whereas `+` adds them up.
    fn join(self, other: GfTranslation<T>) -> GfTranslation<T> {
        GfTranslation {
            var_info: self.var_info.join(&other.var_info),
            gf: self.gf + other.gf,
            rest: self.rest.max(&other.rest),
        }
    }
}

impl<T: Number> std::ops::Add for GfTranslation<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        GfTranslation {
            var_info: self.var_info.join(&other.var_info),
            gf: self.gf + other.gf,
            rest: self.rest + other.rest,
        }
    }
}

impl<T: Number> std::ops::MulAssign<T> for GfTranslation<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.gf *= GenFun::constant(rhs.clone());
        self.rest *= GenFun::constant(rhs);
    }
}

pub struct GfTransformer<T> {
    default_unroll: usize,
    support: SupportTransformer,
    _phantom: PhantomData<T>,
}

impl<T> Default for GfTransformer<T> {
    fn default() -> Self {
        Self {
            default_unroll: DEFAULT_UNROLL,
            support: SupportTransformer,
            _phantom: PhantomData,
        }
    }
}

impl<T: Number> Transformer for GfTransformer<T> {
    type Domain = GfTranslation<T>;

    fn init(&mut self, program: &Program) -> GfTranslation<T> {
        let var_info = self.support.init(program);
        let gf = GenFun::one();
        let rest = GenFun::zero();
        GfTranslation { var_info, gf, rest }
    }

    fn transform_event(
        &mut self,
        event: &Event,
        init: GfTranslation<T>,
    ) -> (GfTranslation<T>, GfTranslation<T>) {
        fn gf_in_set<T: Number>(var: Var, set: &[Natural], gf: GenFun<T>) -> GenFun<T> {
            if let &[order] = &set {
                let order = order.0;
                gf.taylor_coeff_at_zero(var, order as usize) * GenFun::var(var).pow(order)
            } else {
                gf.taylor_polynomial_at_zero(var, set.iter().map(|n| n.0 as usize).collect())
            }
        }
        let var_info = init.var_info.clone();
        let rest = init.rest.clone();
        let gf = init.gf.clone();
        let gf = match event {
            Event::InSet(var, set) => gf_in_set(*var, set, gf),
            Event::VarComparison(v1, comp, v2) => {
                let (scrutinee, other, reversed, range) = match (
                    var_info[v1].finite_nonempty_range(),
                    var_info[v2].finite_nonempty_range(),
                ) {
                    (None, None) => panic!("Cannot compare two variables with infinite support."),
                    (None, Some(r)) => (*v2, *v1, false, r),
                    (Some(r), None) => (*v1, *v2, true, r),
                    (Some(r1), Some(r2)) => {
                        if r1.end() - r1.start() <= r2.end() - r2.start() {
                            (*v1, *v2, true, r1)
                        } else {
                            (*v2, *v1, false, r2)
                        }
                    }
                };
                let mut result = GenFun::zero();
                for i in range {
                    let gf_scrutinee_eq_i = gf_in_set(scrutinee, &[Natural(i)], gf.clone());
                    let summand = match (comp, reversed) {
                        (Comparison::Eq, _) => gf_in_set(other, &[Natural(i)], gf_scrutinee_eq_i),
                        (Comparison::Lt, false) => gf_in_set(
                            other,
                            &(0..i).map(Natural).collect::<Vec<_>>(),
                            gf_scrutinee_eq_i,
                        ),
                        (Comparison::Lt, true) => {
                            gf_scrutinee_eq_i.clone()
                                - gf_in_set(
                                    other,
                                    &(0..=i).map(Natural).collect::<Vec<_>>(),
                                    gf_scrutinee_eq_i,
                                )
                        }
                        (Comparison::Le, false) => gf_in_set(
                            other,
                            &(0..=i).map(Natural).collect::<Vec<_>>(),
                            gf_scrutinee_eq_i,
                        ),
                        (Comparison::Le, true) => {
                            gf_scrutinee_eq_i.clone()
                                - gf_in_set(
                                    other,
                                    &(0..i).map(Natural).collect::<Vec<_>>(),
                                    gf_scrutinee_eq_i,
                                )
                        }
                    };
                    result += summand;
                }
                result
            }
            Event::DataFromDist(data, dist) => {
                if let Some(factor) = event.recognize_const_prob() {
                    GenFun::constant(factor) * gf
                } else {
                    self.transform_data_from_dist(*data, dist, &var_info, gf, rest.clone())
                }
            }
            Event::Complement(e) => {
                let (_, e) = self.transform_event(e, init.clone());
                e.gf
            }
            Event::Intersection(es) => {
                let mut then_result = init.clone();
                for e in es {
                    let (then, _) = self.transform_event(e, then_result);
                    then_result = then;
                }
                then_result.gf
            }
        };
        let (then_info, else_info) = self.support.transform_event(event, var_info);
        (
            GfTranslation {
                var_info: then_info,
                gf: gf.clone(),
                rest: rest.clone(),
            },
            GfTranslation {
                var_info: else_info,
                gf: init.gf - gf,
                rest,
            },
        )
    }

    #[allow(clippy::too_many_lines)]
    fn transform_statement(
        &mut self,
        stmt: &Statement,
        init: GfTranslation<T>,
    ) -> GfTranslation<T> {
        use Statement::*;
        let direct_var_info = if cfg!(debug_assertions) {
            Some(
                self.support
                    .transform_statement(stmt, init.var_info.clone()),
            )
        } else {
            None
        };
        let result = match stmt {
            Sample {
                var: v,
                distribution,
                add_previous_value,
            } => Self::transform_distribution(distribution, *v, init, *add_previous_value),
            Assign {
                var: v,
                add_previous_value,
                addend,
                offset,
            } => {
                let v = *v;
                let mut gf = init.gf;
                let var_info = init.var_info;
                let rest = init.rest;
                let var = GenFun::var(v);
                let mut v_exp = if *add_previous_value { 1 } else { 0 };
                let w_subst = if let Some((factor, w)) = addend {
                    if v == *w {
                        v_exp += factor.0;
                        None
                    } else if var_info[w].is_discrete() {
                        Some((*w, GenFun::var(*w) * var.pow(factor.0)))
                    } else {
                        assert!(
                            !var_info[v].is_discrete() || !*add_previous_value,
                            "cannot add a continuous to a discrete variable"
                        );
                        Some((
                            *w,
                            GenFun::var(*w) + var.clone() * GenFun::from_u32(factor.0),
                        ))
                    }
                } else {
                    None
                };
                if var_info[v].is_discrete() {
                    gf = gf.substitute_var(v, var.pow(v_exp));
                } else {
                    gf = gf.substitute_var(v, var.clone() * GenFun::from_u32(v_exp));
                }
                if let Some((w, w_subst)) = w_subst {
                    gf = gf.substitute_var(w, w_subst);
                }
                let var_info = self.support.transform_statement(stmt, var_info);
                let gf = if var_info[v].is_discrete() {
                    gf * var.pow(offset.0)
                } else {
                    gf * (var * GenFun::from_u32(offset.0)).exp()
                };
                GfTranslation { var_info, gf, rest }
            }
            Decrement { var, offset } => {
                let v = *var;
                let gf = init.gf;
                let var_info = init.var_info;
                let rest = init.rest;
                assert!(
                    var_info[v].is_discrete(),
                    "cannot decrement continuous variables"
                );
                let var_info = self.support.transform_statement(stmt, var_info);
                let gf = gf.shift_down_taylor_at_zero(v, offset.0 as usize);
                GfTranslation { var_info, gf, rest }
            }
            IfThenElse { cond, then, els } => {
                if let Some(factor) = cond.recognize_const_prob::<T>() {
                    // In this case we can avoid path explosion by multiplying with factor
                    // AFTER transforming the if- and else-branches:
                    let mut translation_then = self.transform_statements(then, init.clone());
                    let mut translation_else = self.transform_statements(els, init);
                    translation_then *= factor.clone();
                    translation_else *= T::one() - factor;
                    translation_then + translation_else
                } else {
                    let (then_before, else_before) = self.transform_event(cond, init);
                    let then_after = self.transform_statements(then, then_before);
                    let else_after = self.transform_statements(els, else_before);
                    then_after.join(else_after)
                }
            }
            While { cond, unroll, body } => {
                eprintln!("WARNING: support for while loops is EXPERIMENTAL");
                println!("WARNING: results are APPROXIMATE due to presence of loops: exact inference is only possible for loop-free programs");
                let mut result = GfTranslation::zero(init.var_info.num_vars());
                let var_info = self
                    .support
                    .transform_statement(stmt, init.var_info.clone());
                let mut rest = init;
                for _ in 0..unroll.unwrap_or(self.default_unroll) {
                    let (loop_enter, loop_exit) = self.transform_event(cond, rest);
                    result = result.join(loop_exit);
                    rest = self.transform_statements(&body, loop_enter);
                }
                let new_rest = marginalize_all(rest.gf, &var_info);
                let rest = rest.rest + new_rest;
                GfTranslation {
                    var_info,
                    rest,
                    ..result
                }
            }
            Fail => GfTranslation::zero(init.var_info.num_vars()),
            Normalize {
                given_vars,
                stmts: block,
            } => self.transform_normalize(given_vars, block, init),
        };
        if let Some(direct_var_info) = direct_var_info {
            debug_assert_eq!(
                result.var_info, direct_var_info,
                "inconsistent variable support info for:\n{stmt}"
            );
        }
        result
    }
}

impl<T: Number> GfTransformer<T> {
    pub fn with_default_unroll(mut self, default_unroll: Option<usize>) -> Self {
        self.default_unroll = default_unroll.unwrap_or(DEFAULT_UNROLL);
        self
    }

    fn compound_dist(
        gf: &GenFun<T>,
        base: &GenFun<T>,
        sampled_var: Var,
        param_var: Var,
        add_previous_value: bool,
        param_var_discrete: bool,
        subst: GenFun<T>,
    ) -> GenFun<T> {
        if sampled_var == param_var {
            if add_previous_value {
                let substitution = if param_var_discrete {
                    GenFun::var(param_var) * subst
                } else {
                    GenFun::var(param_var) + subst
                };
                gf.substitute_var(param_var, substitution)
            } else {
                gf.substitute_var(param_var, subst)
            }
        } else {
            let substitution = if param_var_discrete {
                GenFun::var(param_var) * subst
            } else {
                GenFun::var(param_var) + subst
            };
            base.substitute_var(param_var, substitution)
        }
    }

    #[allow(clippy::too_many_lines)]
    fn transform_distribution(
        dist: &Distribution,
        v: Var,
        translation: GfTranslation<T>,
        add_previous_value: bool,
    ) -> GfTranslation<T> {
        use Distribution::*;
        let base = if add_previous_value {
            translation.gf.clone()
        } else {
            marginalize_out(v, &translation.gf, &translation.var_info)
        };
        let new_var_info = SupportTransformer::transform_distribution(
            dist,
            v,
            translation.var_info.clone(),
            add_previous_value,
        );
        let gf = &translation.gf;
        let gf = match dist {
            Dirac(a) => {
                let dirac = if let Some(a) = a.as_integer() {
                    GenFun::var(v).pow(a)
                } else {
                    (GenFun::var(v) * GenFun::from_ratio(*a)).exp()
                };
                dirac * base
            }
            Bernoulli(p) => {
                let bernoulli =
                    GenFun::from_ratio(*p) * GenFun::var(v) + GenFun::from_ratio(p.complement());
                bernoulli * base
            }
            BernoulliVarProb(w) => {
                let prob_times_gf = if translation.var_info[w].is_discrete() {
                    gf.derive(*w, 1) * GenFun::var(*w)
                } else {
                    gf.derive(*w, 1)
                };
                let prob_times_base = if add_previous_value {
                    prob_times_gf
                } else {
                    marginalize_out(v, &prob_times_gf, &translation.var_info)
                };
                let v_term = if new_var_info[v].is_discrete() {
                    GenFun::var(v)
                } else {
                    GenFun::var(v).exp()
                };
                base + (v_term - GenFun::one()) * prob_times_base
            }
            BinomialVarTrials(w, p) => {
                let subst =
                    GenFun::from_ratio(*p) * GenFun::var(v) + GenFun::from_ratio(p.complement());
                Self::compound_dist(gf, &base, v, *w, add_previous_value, true, subst)
            }
            Binomial(n, p) => {
                let binomial = (GenFun::from_ratio(*p) * GenFun::var(v)
                    + GenFun::from_ratio(p.complement()))
                .pow(n.0);
                binomial * base
            }
            Categorical(rs) => {
                let mut categorical = GenFun::zero();
                for r in rs.iter().rev() {
                    categorical *= GenFun::var(v);
                    categorical += GenFun::from_ratio(*r);
                }
                categorical * base
            }
            NegBinomialVarSuccesses(w, p) => {
                let subst = GenFun::from_ratio(*p)
                    / (GenFun::one() - GenFun::from_ratio(p.complement()) * GenFun::var(v));
                Self::compound_dist(gf, &base, v, *w, add_previous_value, true, subst)
            }
            NegBinomial(n, p) => {
                let geometric = GenFun::from_ratio(*p)
                    / (GenFun::one() - GenFun::from_ratio(p.complement()) * GenFun::var(v));
                geometric.pow(n.0) * base
            }
            Geometric(p) => {
                let geometric = GenFun::from_ratio(*p)
                    / (GenFun::one() - GenFun::from_ratio(p.complement()) * GenFun::var(v));
                geometric * base
            }
            Poisson(lambda) => {
                let poisson =
                    (GenFun::from_ratio(*lambda) * (GenFun::var(v) - GenFun::one())).exp();
                poisson * base
            }
            PoissonVarRate(lambda, w) => {
                let w_discrete = translation.var_info[w].is_discrete();
                let subst = if w_discrete {
                    (GenFun::from_ratio(*lambda) * (GenFun::var(v) - GenFun::one())).exp()
                } else {
                    GenFun::from_ratio(*lambda) * (GenFun::var(v) - GenFun::one())
                };
                Self::compound_dist(gf, &base, v, *w, add_previous_value, w_discrete, subst)
            }
            Uniform { start, end } => {
                let mut uniform = GenFun::zero();
                assert!(end.0 > start.0, "Uniform distribution cannot have length 0");
                let length = end.0 - start.0;
                let weight = GenFun::from_ratio(PosRatio::new(1, u64::from(length)));
                for _ in 0..length {
                    uniform = weight.clone() + GenFun::var(v) * uniform;
                }
                uniform *= GenFun::var(v).pow(start.0);
                uniform * base
            }
            Exponential { rate } => {
                let beta = GenFun::from_ratio(*rate);
                let exponential = beta.clone() / (beta - GenFun::var(v));
                exponential * base
            }
            Gamma { shape, rate } => {
                let beta = GenFun::from_ratio(*rate);
                let gamma = if let Some(shape) = shape.as_integer() {
                    // Optimized representation avoiding logarithms for integer exponents
                    (beta.clone() / (beta - GenFun::var(v))).pow(shape)
                } else {
                    (GenFun::from_ratio(*shape) * (beta.log() - (beta - GenFun::var(v)).log()))
                        .exp()
                };
                gamma * base
            }
            UniformCont { start, end } => {
                let width =
                    T::from_ratio(end.numer, end.denom) - T::from_ratio(start.numer, start.denom);
                let x = GenFun::constant(width) * GenFun::var(v);
                let uniform =
                    GenFun::uniform_mgf(x) * (GenFun::from_ratio(*start) * GenFun::var(v)).exp();
                uniform * base
            }
        };
        GfTranslation {
            gf,
            var_info: new_var_info,
            rest: translation.rest,
        }
    }

    fn transform_data_from_dist(
        &mut self,
        data: Natural,
        dist: &Distribution,
        var_info: &VarSupport,
        gf: GenFun<T>,
        rest: GenFun<T>,
    ) -> GenFun<T> {
        match dist {
            Distribution::BernoulliVarProb(var) => {
                let prob_times_gf = if var_info[var].is_discrete() {
                    gf.derive(*var, 1) * GenFun::var(*var)
                } else {
                    gf.derive(*var, 1)
                };
                match data.0 {
                    0 => gf - prob_times_gf,
                    1 => prob_times_gf,
                    _ => GenFun::zero(),
                }
            }
            Distribution::BinomialVarTrials(var, p) => {
                let order = data.0 as usize;
                let replacement = GenFun::from_ratio(p.complement()) * GenFun::var(*var);
                gf.taylor_coeff(*var, order)
                    .substitute_var(*var, replacement)
                    * (GenFun::from_ratio(*p) * GenFun::var(*var)).pow(data.0)
            }
            _ => {
                // TODO: this can be optimized for distributions that only have constant parameters.
                // In that case, we should just multiply by the probability mass function.
                let new_var = Var(gf.used_vars().num_vars());
                let sample_stmt = Statement::Sample {
                    var: new_var,
                    distribution: dist.clone(),
                    add_previous_value: false,
                };
                let translation = GfTranslation {
                    var_info: var_info.clone(),
                    gf,
                    rest,
                };
                let new_translation = self.transform_statement(&sample_stmt, translation);
                let gf = new_translation
                    .gf
                    .taylor_coeff_at_zero(new_var, data.0 as usize);
                marginalize_out(new_var, &gf, &new_translation.var_info)
            }
        }
    }

    fn transform_normalize(
        &mut self,
        given_vars: &[Var],
        block: &[Statement],
        translation: GfTranslation<T>,
    ) -> GfTranslation<T> {
        if given_vars.is_empty() {
            let total_before = marginalize_all(translation.gf.clone(), &translation.var_info);
            let rest_before = translation.rest.clone();
            let translation = self.transform_statements(block, translation);
            let total_after = marginalize_all(translation.gf.clone(), &translation.var_info);
            let rest_after = translation.rest.clone();
            let min_factor = total_before.clone() / (total_after.clone() + rest_after);
            let max_factor = (total_before + rest_before) / total_after;
            GfTranslation {
                var_info: translation.var_info,
                gf: min_factor * translation.gf,
                rest: max_factor * translation.rest,
            }
        } else {
            let v = given_vars[0];
            let rest = &given_vars[1..];
            let support = translation.var_info[v].clone();
            let range = support.finite_nonempty_range().unwrap_or_else(|| panic!("Cannot normalize with respect to variable `{v}`, because its value could not be proven to be bounded."));
            let mut joined = GfTranslation::zero(translation.var_info.num_vars());
            for i in range {
                let summand =
                    translation.gf.taylor_coeff_at_zero(v, i as usize) * GenFun::var(v).pow(i);
                let mut new_var_info = translation.var_info.clone();
                new_var_info.set(v, SupportSet::from(i));
                let summand = GfTranslation {
                    var_info: new_var_info,
                    gf: summand,
                    rest: translation.rest.clone(),
                };
                let result = self.transform_normalize(rest, block, summand);
                joined = joined.join(result);
            }
            joined
        }
    }
}

fn marginalize_out<T: Number>(v: Var, gf: &GenFun<T>, var_info: &VarSupport) -> GenFun<T> {
    if v.id() >= var_info.num_vars() {
        // This can only be a temporary variable of index n, where n is the number of variables.
        // This is introduced for observe c ~ D(X_i) statements for the temporary variable X_n ~ D(X_i).
        assert!(v.id() == var_info.num_vars());
        // In this case, the variable is discrete because we only support discrete observations.
        gf.substitute_var(v, GenFun::one())
    } else if var_info[v].is_discrete() {
        gf.substitute_var(v, GenFun::one())
    } else {
        gf.substitute_var(v, GenFun::zero())
    }
}

fn marginalize_all<T: Number>(gf: GenFun<T>, var_info: &VarSupport) -> GenFun<T> {
    let mut result = gf;
    for v in 0..var_info.num_vars() {
        result = marginalize_out(Var(v), &result, var_info);
    }
    result
}

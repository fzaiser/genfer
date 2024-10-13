use ndarray::{ArrayD, Axis, Slice};
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;
use std::ops::AddAssign;

use crate::{
    numbers::Rational,
    ppl::Var,
    sym_expr::SymExpr,
    util::{binomial, polylog_neg},
};

/// Eventually Geometric Distribution (EGD).
///
/// The initial block is given by `block` and the decay rates by `decays`.
#[derive(Debug, Clone, PartialEq)]
pub struct Egd {
    pub block: ArrayD<SymExpr>,
    pub decays: Vec<SymExpr>,
}

impl Egd {
    pub(crate) fn zero(n: usize) -> Self {
        Egd {
            block: ArrayD::zeros(vec![1; n]),
            decays: vec![SymExpr::zero(); n],
        }
    }

    pub(crate) fn extend_axis(&mut self, var: Var, new_len: usize) {
        let axis = Axis(var.id());
        let old_len = self.block.len_of(axis);
        if new_len <= old_len {
            return;
        }
        let mut new_shape = self.block.shape().to_owned();
        new_shape[var.id()] = new_len - old_len;
        self.block
            .append(axis, ArrayD::zeros(new_shape).view())
            .unwrap();
        for i in old_len..new_len {
            let (left, mut right) = self.block.view_mut().split_at(axis, i);
            right.index_axis_mut(axis, 0).assign(
                &left
                    .index_axis(axis, i - 1)
                    .map(|e| e.clone() * self.decays[var.id()].clone()),
            );
        }
    }

    pub(crate) fn shift_left(&mut self, Var(v): Var, offset: usize) {
        self.extend_axis(Var(v), offset + 2);
        let zero_elem = self
            .block
            .slice_axis(Axis(v), Slice::from(0..=offset))
            .sum_axis(Axis(v));
        self.block
            .slice_axis_inplace(Axis(v), Slice::from(offset..));
        self.block.index_axis_mut(Axis(v), 0).assign(&zero_elem);
    }

    pub(crate) fn shift_right(&mut self, Var(v): Var, offset: usize) {
        let mut zero_shape = self.block.shape().to_owned();
        zero_shape[v] = offset;
        self.block = ndarray::concatenate(
            Axis(v),
            &[ArrayD::zeros(zero_shape).view(), self.block.view()],
        )
        .unwrap();
    }

    pub(crate) fn add_categorical(&mut self, Var(v): Var, categorical: &[Rational]) {
        let len = self.block.len_of(Axis(v));
        let max = categorical.len() - 1;
        self.extend_axis(Var(v), len + max);
        let mut new_shape = self.block.shape().to_owned();
        new_shape[v] = len + max;
        let block = self.block.clone();
        self.block = ArrayD::zeros(new_shape);
        for (offset, prob) in categorical.iter().enumerate() {
            if prob > &Rational::zero() {
                self.block
                    .slice_axis_mut(Axis(v), Slice::from(offset..len + max))
                    .add_assign(
                        &block
                            .slice_axis(Axis(v), Slice::from(0..len + max - offset))
                            .map(|e| e.clone() * SymExpr::from(prob.clone())),
                    );
            }
        }
    }

    pub(crate) fn marginalize_out(&self, var: Var) -> Self {
        let len = self.block.len_of(Axis(var.id()));
        let axis = Axis(var.id());
        let mut decays = self.decays.clone();
        let mut block = self
            .block
            .slice_axis(axis, Slice::from(len - 1..len))
            .to_owned();
        block.map_inplace(|e| *e /= SymExpr::one() - decays[var.id()].clone());
        for subview in self.block.axis_chunks_iter(axis, 1).take(len - 1) {
            block += &subview;
        }
        decays[var.id()] = SymExpr::zero();
        Self { block, decays }
    }

    pub(crate) fn marginal(&self, var: Var) -> Self {
        let mut result = self.clone();
        for v in 0..self.block.ndim() {
            if Var(v) != var {
                result = result.marginalize_out(Var(v));
            }
        }
        result
    }

    pub fn resolve(&self, assignments: &[Rational]) -> Egd {
        let cache = &mut FxHashMap::default();
        let block = self
            .block
            .map(|c| SymExpr::from(c.eval_exact(assignments, cache)));
        let decays = self
            .decays
            .iter()
            .map(|p| SymExpr::from(p.eval_exact(assignments, cache)))
            .collect();
        Self { block, decays }
    }

    pub(crate) fn moments(&self, Var(v): Var, limit: usize) -> Vec<SymExpr> {
        assert!(self.block.ndim() > v);
        assert!(
            self.block
                .shape()
                .iter()
                .enumerate()
                .all(|(i, &len)| { i == v || len == 1 }),
            "Bound should be marginalized before computing moments"
        );
        let q = self.decays[v].clone();
        let initial = self.block.iter().cloned().collect::<Vec<_>>();
        Self::moments_of(&initial, q, limit)
    }

    pub fn moments_exact(&self, Var(v): Var, limit: usize) -> Vec<Rational> {
        assert!(self.block.ndim() > v);
        assert!(
            self.block
                .shape()
                .iter()
                .enumerate()
                .all(|(i, &len)| { i == v || len == 1 }),
            "Bound should be marginalized before computing moments"
        );
        let q = self.decays[v].extract_constant().unwrap().rat();
        let initial = self
            .block
            .iter()
            .map(|e| e.extract_constant().unwrap().rat())
            .collect::<Vec<_>>();
        Self::moments_of(&initial, q, limit)
    }

    fn moments_of<T>(initial: &[T], q: T, limit: usize) -> Vec<T>
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
        let len = initial.len();
        let mut moments = vec![T::zero(); limit];
        for (j, initial_prob) in initial.iter().enumerate().take(len - 1) {
            let mut power = initial_prob.clone();
            for moment in &mut moments {
                *moment += power.clone();
                power *= T::from(j as u64);
            }
        }
        let d = len - 1;
        // The moments of `Geometric(1 - q) / q`:
        let mut geo_moments = polylog_neg(limit as u64, q.clone());

        // We have to adjust the 0-th moment because the polylog function Li_{-n}(x) is defined
        // as an infinite sum starting at 1 instead of 0.
        // This does not make a difference because the 0-th summand is 0, except for k = 0.
        geo_moments[0] += T::one();

        let binom = binomial(limit as u64);
        let d_powers =
            std::iter::successors(Some(T::one()), |p| Some(p.clone() * T::from(d as u64)))
                .take(limit)
                .collect::<Vec<_>>();

        for k in 0..limit {
            for i in 0..=k {
                moments[k] += T::from(binom[k][i])
                    * d_powers[k - i].clone()
                    * initial[d].clone()
                    * geo_moments[i].clone();
            }
        }

        moments
    }

    pub fn probs_exact(&self, Var(v): Var, limit: usize) -> Vec<Rational> {
        let decay = self.decays[v].extract_constant().unwrap().rat();
        let mut probs = self
            .block
            .iter()
            .map(|e| e.extract_constant().unwrap().rat())
            .collect::<Vec<_>>();
        let mut last = probs[probs.len() - 1].clone();
        for _ in probs.len()..limit {
            last *= decay.clone();
            probs.push(last.clone());
        }
        probs
    }

    pub fn total_mass(&self) -> SymExpr {
        let mut res = self.clone();
        for v in 0..res.block.ndim() {
            res = res.marginalize_out(Var(v));
        }
        res.block.first().unwrap().clone()
    }

    pub fn expected_value(&self, v: Var) -> SymExpr {
        self.marginal(v).moments(v, 2)[1].clone()
    }

    pub fn tail_objective(&self, v: Var) -> SymExpr {
        (SymExpr::one() - self.decays[v.id()].clone()).inverse()
    }
}

impl std::fmt::Display for Egd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "EGD(\n{},", self.block)?;
        write!(f, "[")?;
        for param in &self.decays {
            write!(f, "{param}, ")?;
        }
        write!(f, "])")
    }
}

impl std::ops::Mul<SymExpr> for Egd {
    type Output = Self;

    fn mul(mut self, rhs: SymExpr) -> Self::Output {
        for elem in &mut self.block {
            *elem *= rhs.clone();
        }
        self
    }
}

impl std::ops::MulAssign<SymExpr> for Egd {
    fn mul_assign(&mut self, rhs: SymExpr) {
        for elem in &mut self.block {
            *elem *= rhs.clone();
        }
    }
}

impl std::ops::Div<SymExpr> for Egd {
    type Output = Self;

    fn div(mut self, rhs: SymExpr) -> Self::Output {
        for elem in &mut self.block {
            *elem /= rhs.clone();
        }
        self
    }
}

impl std::ops::DivAssign<SymExpr> for Egd {
    fn div_assign(&mut self, rhs: SymExpr) {
        for elem in &mut self.block {
            *elem /= rhs.clone();
        }
    }
}

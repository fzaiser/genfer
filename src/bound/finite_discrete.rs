use ndarray::{ArrayD, Axis, Slice};
use num_traits::Zero;
use std::ops::AddAssign;

use crate::{numbers::Rational, ppl::Var};

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteDiscrete {
    pub masses: ArrayD<Rational>,
}

impl FiniteDiscrete {
    pub(crate) fn zero(n: usize) -> FiniteDiscrete {
        FiniteDiscrete {
            masses: ArrayD::zeros(vec![1; n]),
        }
    }

    pub(crate) fn marginalize_out(&self, Var(v): Var) -> FiniteDiscrete {
        let masses = self.masses.sum_axis(Axis(v)).insert_axis(Axis(v));
        FiniteDiscrete { masses }
    }

    pub fn total_mass(&self) -> Rational {
        self.masses.sum()
    }

    pub(crate) fn extend_axis(&mut self, Var(v): Var, new_len: usize) {
        let axis = Axis(v);
        let old_len = self.masses.len_of(axis);
        if new_len <= old_len {
            return;
        }
        let mut new_shape = self.masses.shape().to_owned();
        new_shape[v] = new_len - old_len;
        self.masses
            .append(axis, ArrayD::zeros(new_shape).view())
            .unwrap();
    }

    pub(crate) fn shift_left(&mut self, Var(v): Var, offset: usize) {
        let len = self.masses.len_of(Axis(v));
        let zero_elem = self
            .masses
            .slice_axis(Axis(v), Slice::from(0..=offset.min(len - 1)))
            .sum_axis(Axis(v));
        self.masses
            .slice_axis_inplace(Axis(v), Slice::from(offset.min(len - 1)..));
        self.masses.index_axis_mut(Axis(v), 0).assign(&zero_elem);
    }

    pub(crate) fn shift_right(&mut self, Var(v): Var, offset: usize) {
        let mut zero_shape = self.masses.shape().to_owned();
        zero_shape[v] = offset;
        self.masses = ndarray::concatenate(
            Axis(v),
            &[ArrayD::zeros(zero_shape).view(), self.masses.view()],
        )
        .unwrap();
    }

    pub(crate) fn add_categorical(&mut self, Var(v): Var, categorical: &[Rational]) {
        let len = self.masses.len_of(Axis(v));
        let max = categorical.len() - 1;
        self.extend_axis(Var(v), len + max);
        let mut new_shape = self.masses.shape().to_owned();
        new_shape[v] = len + max;
        let masses = self.masses.clone();
        self.masses = ArrayD::zeros(new_shape);
        for (offset, prob) in categorical.iter().enumerate() {
            if prob > &Rational::zero() {
                self.masses
                    .slice_axis_mut(Axis(v), Slice::from(offset..len + offset))
                    .add_assign(
                        &masses
                            .slice_axis(Axis(v), Slice::from(0..len))
                            .map(|e| e.clone() * prob.clone()),
                    );
            }
        }
    }

    pub fn probs(&self, Var(v): Var) -> Vec<Rational> {
        self.masses.axis_iter(Axis(v)).map(|ms| ms.sum()).collect()
    }

    pub fn moments(&self, Var(v): Var, n: usize) -> Vec<Rational> {
        let probs = self.probs(Var(v));
        let mut moments = vec![Rational::zero(); n];
        for (k, p) in probs.into_iter().enumerate() {
            let mut power = p;
            for moment in &mut moments {
                *moment += power.clone();
                power *= Rational::from_int(k);
            }
        }
        moments
    }
}

impl std::fmt::Display for FiniteDiscrete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.masses.map(Rational::to_f64).fmt(f)
    }
}

impl std::ops::AddAssign<&Self> for FiniteDiscrete {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.masses.ndim(), rhs.masses.ndim());
        if self.masses.shape() == rhs.masses.shape() {
            self.masses += &rhs.masses;
        } else {
            let lhs_shape = self.masses.shape();
            let rhs_shape = rhs.masses.shape();
            let shape = lhs_shape
                .iter()
                .zip(rhs_shape)
                .map(|(&l, &r)| l.max(r))
                .collect::<Vec<_>>();
            let mut masses = ArrayD::zeros(shape);
            masses
                .slice_each_axis_mut(|ax| (0..lhs_shape[ax.axis.index()]).into())
                .add_assign(&self.masses);
            masses
                .slice_each_axis_mut(|ax| (0..rhs_shape[ax.axis.index()]).into())
                .add_assign(&rhs.masses);
            self.masses = masses;
        }
    }
}

impl std::ops::Add for &FiniteDiscrete {
    type Output = FiniteDiscrete;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.masses.ndim(), rhs.masses.ndim());
        let lhs_shape = self.masses.shape();
        let rhs_shape = rhs.masses.shape();
        let shape = lhs_shape
            .iter()
            .zip(rhs_shape)
            .map(|(&l, &r)| l.max(r))
            .collect::<Vec<_>>();
        let mut masses = ArrayD::zeros(shape);
        masses
            .slice_each_axis_mut(|ax| (0..lhs_shape[ax.axis.index()]).into())
            .add_assign(&self.masses);
        masses
            .slice_each_axis_mut(|ax| (0..rhs_shape[ax.axis.index()]).into())
            .add_assign(&rhs.masses);
        FiniteDiscrete { masses }
    }
}

impl std::ops::MulAssign<Rational> for FiniteDiscrete {
    fn mul_assign(&mut self, rhs: Rational) {
        for elem in &mut self.masses {
            *elem *= rhs.clone();
        }
    }
}

impl std::ops::Mul<Rational> for FiniteDiscrete {
    type Output = Self;

    fn mul(mut self, rhs: Rational) -> Self::Output {
        self *= rhs;
        self
    }
}

impl std::ops::DivAssign<Rational> for FiniteDiscrete {
    fn div_assign(&mut self, rhs: Rational) {
        for elem in &mut self.masses {
            *elem /= rhs.clone();
        }
    }
}

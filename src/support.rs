use std::ops::{Range, RangeFrom, RangeInclusive};

use num_traits::Zero;

use crate::{
    interval::Interval,
    number::{FloatNumber, IntervalNumber, Number, Rational},
};

/// Support set of a random variable (overapproximated as a range)
#[derive(Clone, Debug, PartialEq)]
pub enum SupportSet {
    Empty,
    Range { start: u32, end: Option<u32> }, // TODO: should be ExtendedNat
    Interval { start: Rational, end: Rational },
}
impl SupportSet {
    pub fn empty() -> Self {
        Self::Empty
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn is_zero(&self) -> bool {
        matches!(
            self,
            Self::Range {
                start: 0,
                end: Some(0)
            }
        )
    }

    pub fn zero() -> Self {
        Self::Range {
            start: 0,
            end: Some(0),
        }
    }

    pub fn point(x: u32) -> Self {
        Self::Range {
            start: x,
            end: Some(x),
        }
    }

    pub fn naturals() -> Self {
        Self::Range {
            start: 0,
            end: None,
        }
    }

    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Empty, x) | (x, Self::Empty) => x.clone(),
            (
                Self::Range { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => Self::Range {
                start: (*start).min(*start2),
                end: match (end, end2) {
                    (Some(x), Some(y)) => Some((*x).max(*y)),
                    _ => None,
                },
            },
            (
                Self::Interval { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: start.min(start2),
                end: end.max(end2),
            },
            (
                Self::Range { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: Rational::from(*start).min(start2),
                end: if let Some(end) = end {
                    Rational::from(*end).max(end2)
                } else {
                    Rational::infinity()
                },
            },
            (
                Self::Interval { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: start.min(&Rational::from(*start2)),
                end: if let Some(end2) = end2 {
                    end.max(&Rational::from(*end2))
                } else {
                    Rational::infinity()
                },
            },
        }
    }

    pub fn saturating_sub(&self, other: u32) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Range { start, end } => Self::Range {
                start: start.saturating_sub(other),
                end: end.map(|x| x.saturating_sub(other)),
            },
            Self::Interval { start, end } => Self::Interval {
                start: (start.clone() - other.into()).max(&Rational::zero()),
                end: (end.clone() - other.into()).max(&Rational::zero()),
            },
        }
    }

    pub fn finite_nonempty_range(&self) -> Option<std::ops::RangeInclusive<u32>> {
        match self {
            Self::Empty | Self::Interval { .. } => None,
            Self::Range { start, end } => Some(*start..=(*end)?),
        }
    }

    pub fn is_discrete(&self) -> bool {
        match self {
            Self::Empty | Self::Range { .. } => true,
            Self::Interval { .. } => false,
        }
    }

    pub fn interval(start: Rational, end: Rational) -> Self {
        if start > end {
            return Self::empty();
        }
        Self::Interval { start, end }
    }

    pub fn nonneg_reals() -> SupportSet {
        Self::interval(Rational::zero(), Rational::infinity())
    }

    pub fn is_subset_of(&self, other: &SupportSet) -> bool {
        match (self, other) {
            (Self::Empty, _) => true,
            (_, Self::Empty) | (Self::Interval { .. }, Self::Range { .. }) => false,
            (
                Self::Range { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => start >= start2 && (end2.is_none() || (end.is_some() && end <= end2)),
            (
                Self::Interval { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => start >= start2 && end <= end2,
            (
                Self::Range { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => {
                &Rational::from(*start) >= start2
                    && end.is_some()
                    && &Rational::from(end.unwrap()) <= end2
            }
        }
    }

    pub fn vec_is_subset_of(lhs: &[Self], rhs: &[Self]) -> bool {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter().zip(rhs.iter()).all(|(x, y)| x.is_subset_of(y))
    }

    #[inline]
    pub fn retain_only(&mut self, set: impl Iterator<Item = u32>) {
        let mut set = set.collect::<Vec<_>>();
        set.sort_unstable();
        *self = self.retain_only_impl(&set);
    }

    fn retain_only_impl(&self, set: &[u32]) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Range { start, end } => {
                let mut new_start = None;
                let mut new_end = None;
                for v in set {
                    if start <= v && v <= &end.unwrap_or(u32::MAX) {
                        if new_start.is_none() {
                            new_start = Some(*v);
                        }
                        new_end = Some(*v);
                    }
                }
                if let (Some(start), end) = (new_start, new_end) {
                    Self::Range { start, end }
                } else {
                    Self::empty()
                }
            }
            Self::Interval { .. } => self.clone(),
        }
    }

    #[inline]
    pub fn remove_all(&mut self, set: impl Iterator<Item = u32>) {
        let mut set = set.collect::<Vec<_>>();
        set.sort_unstable();
        self.remove_all_impl(&set);
    }

    fn remove_all_impl(&mut self, set: &[u32]) {
        match self {
            Self::Empty | Self::Interval { .. } => {}
            Self::Range { start, end } => {
                if set.is_empty() {
                    return;
                }
                for v in set {
                    if v == start {
                        *start = v + 1;
                    }
                }
                if let Some(end) = end {
                    for v in set.iter().rev() {
                        if v == end {
                            if v == &0 {
                                *end = 0;
                                *start = 1; // to ensure *end < *start --> result is set to empty later
                            } else {
                                *end = v - 1;
                            }
                        }
                    }
                }
                if *start > end.unwrap_or(u32::MAX) {
                    *self = Self::Empty;
                }
            }
        }
    }

    pub fn to_interval<T: IntervalNumber>(&self) -> Option<Interval<T>> {
        match self {
            Self::Empty => None,
            Self::Range { start, end } => Some(Interval::exact(
                T::from(*start),
                end.map_or(T::infinity(), T::from),
            )),
            Self::Interval { start, end } => {
                let (start_n, start_d) = start.to_ratio().unwrap();
                let start = T::from_ratio(start_n.try_into().unwrap(), start_d.try_into().unwrap());
                let end = if end.is_infinite() {
                    T::infinity()
                } else {
                    let (end_n, end_d) = end.to_ratio().unwrap();
                    T::from_ratio(end_n.try_into().unwrap(), end_d)
                };
                Some(Interval::exact(start, end))
            }
        }
    }

    pub fn contains(&self, i: u32) -> bool {
        match self {
            Self::Empty => false,
            Self::Range { start, end } => i >= *start && end.map_or(true, |end| i <= end),
            Self::Interval { start, end } => {
                let i = Rational::from_int(i);
                &i >= start && &i <= end
            }
        }
    }
}

impl From<u32> for SupportSet {
    fn from(x: u32) -> Self {
        Self::Range {
            start: x,
            end: Some(x),
        }
    }
}

impl From<Range<u32>> for SupportSet {
    fn from(range: Range<u32>) -> Self {
        if range.end <= range.start {
            return Self::empty();
        }
        Self::Range {
            start: range.start,
            end: Some(range.end - 1),
        }
    }
}

impl From<Range<Rational>> for SupportSet {
    fn from(range: Range<Rational>) -> Self {
        if range.end <= range.start {
            return Self::empty();
        }
        Self::Interval {
            start: range.start,
            end: range.end,
        }
    }
}

impl From<RangeInclusive<u32>> for SupportSet {
    fn from(range: RangeInclusive<u32>) -> Self {
        let (start, end) = range.into_inner();
        if start > end {
            return Self::empty();
        }
        Self::Range {
            start,
            end: Some(end),
        }
    }
}

impl From<RangeFrom<u32>> for SupportSet {
    fn from(range: RangeFrom<u32>) -> Self {
        Self::Range {
            start: range.start,
            end: None,
        }
    }
}

impl std::fmt::Display for SupportSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "∅"),
            Self::Range { start, end } => {
                if let Some(end) = end {
                    if start == end {
                        return write!(f, "{{{start}}}");
                    }
                    write!(f, "{{{start}, ..., {end}}}")
                } else {
                    write!(f, "{{{start}, ...}}")
                }
            }
            Self::Interval { start, end } => {
                if end == &Rational::infinity() {
                    write!(f, "[{start}, ∞)")
                } else {
                    write!(f, "[{start}, {end}]")
                }
            }
        }
    }
}

impl std::ops::Add for SupportSet {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Empty, x) | (x, Self::Empty) => x,
            (
                Self::Range { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => Self::Range {
                start: start.saturating_add(start2),
                end: match (end, end2) {
                    (Some(x), Some(y)) => x.checked_add(y),
                    _ => None,
                },
            },
            (
                Self::Interval { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: start + start2,
                end: end + end2,
            },
            (
                Self::Range { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: Rational::from(start) + start2,
                end: if let Some(end) = end {
                    Rational::from(end) + end2
                } else {
                    Rational::infinity()
                },
            },
            (
                Self::Interval { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: start + Rational::from(start2),
                end: if let Some(end2) = end2 {
                    end + Rational::from(end2)
                } else {
                    Rational::infinity()
                },
            },
        }
    }
}

impl std::ops::AddAssign for SupportSet {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Mul<u32> for SupportSet {
    type Output = Self;

    fn mul(self, rhs: u32) -> Self::Output {
        match self {
            Self::Empty => Self::Empty,
            Self::Range { start, end } => Self::Range {
                start: start * rhs,
                end: end.map(|x| x * rhs),
            },
            Self::Interval { start, end } => Self::Interval {
                start: start * rhs.into(),
                end: end * rhs.into(),
            },
        }
    }
}

impl std::ops::Mul<SupportSet> for SupportSet {
    type Output = Self;

    fn mul(self, rhs: SupportSet) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::from(0);
        }
        match (self, rhs) {
            (Self::Empty, x) | (x, Self::Empty) => x,
            (
                Self::Range { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => Self::Range {
                start: start * start2,
                end: match (end, end2) {
                    (Some(x), Some(y)) => Some(x * y),
                    _ => None,
                },
            },
            (
                Self::Interval { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: start * start2,
                end: end * end2,
            },
            (
                Self::Range { start, end },
                Self::Interval {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: Rational::from(start) * start2,
                end: if let Some(end) = end {
                    Rational::from(end) * end2
                } else {
                    Rational::infinity()
                },
            },
            (
                Self::Interval { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => Self::Interval {
                start: start * start2.into(),
                end: if let Some(end2) = end2 {
                    end * end2.into()
                } else {
                    Rational::infinity()
                },
            },
        }
    }
}

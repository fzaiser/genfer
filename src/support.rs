use std::ops::{Range, RangeFrom, RangeInclusive};

use crate::{interval::Interval, numbers::Rational};

/// Support set of a random variable (overapproximated as a range)
#[derive(Clone, Debug, PartialEq)]
pub enum SupportSet {
    Empty,
    Range { start: u64, end: Option<u64> },
}
impl SupportSet {
    pub(crate) fn empty() -> Self {
        Self::Empty
    }

    pub(crate) fn is_empty(&self) -> bool {
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

    pub(crate) fn zero() -> Self {
        Self::Range {
            start: 0,
            end: Some(0),
        }
    }

    pub(crate) fn point(x: u64) -> Self {
        Self::Range {
            start: x,
            end: Some(x),
        }
    }

    pub(crate) fn naturals() -> Self {
        Self::Range {
            start: 0,
            end: None,
        }
    }

    pub(crate) fn join(&self, other: &Self) -> Self {
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
        }
    }

    pub(crate) fn saturating_sub(&self, other: u64) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Range { start, end } => Self::Range {
                start: start.saturating_sub(other),
                end: end.map(|x| x.saturating_sub(other)),
            },
        }
    }

    pub fn finite_nonempty_range(&self) -> Option<std::ops::RangeInclusive<u64>> {
        match self {
            Self::Empty => None,
            Self::Range { start, end } => Some(*start..=(*end)?),
        }
    }

    pub(crate) fn is_subset_of(&self, other: &SupportSet) -> bool {
        match (self, other) {
            (Self::Empty, _) => true,
            (_, Self::Empty) => false,
            (
                Self::Range { start, end },
                Self::Range {
                    start: start2,
                    end: end2,
                },
            ) => start >= start2 && (end2.is_none() || (end.is_some() && end <= end2)),
        }
    }

    #[inline]
    pub(crate) fn retain_only(&mut self, set: impl Iterator<Item = u64>) {
        let mut set = set.collect::<Vec<_>>();
        set.sort_unstable();
        *self = self.retain_only_impl(&set);
    }

    fn retain_only_impl(&self, set: &[u64]) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Range { start, end } => {
                let mut new_start = None;
                let mut new_end = None;
                for v in set {
                    if start <= v && v <= &end.unwrap_or(u64::MAX) {
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
        }
    }

    #[inline]
    pub(crate) fn remove_all(&mut self, set: impl Iterator<Item = u64>) {
        let mut set = set.collect::<Vec<_>>();
        set.sort_unstable();
        self.remove_all_impl(&set);
    }

    fn remove_all_impl(&mut self, set: &[u64]) {
        match self {
            Self::Empty => {}
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
                if *start > end.unwrap_or(u64::MAX) {
                    *self = Self::Empty;
                }
            }
        }
    }

    pub fn to_interval(&self) -> Option<Interval<Rational>> {
        match self {
            Self::Empty => None,
            Self::Range { start, end } => Some(Interval::exact(
                Rational::from(*start),
                end.map_or(Rational::infinity(), Rational::from),
            )),
        }
    }

    pub fn contains(&self, i: u64) -> bool {
        match self {
            Self::Empty => false,
            Self::Range { start, end } => i >= *start && end.map_or(true, |end| i <= end),
        }
    }
}

impl From<u64> for SupportSet {
    fn from(x: u64) -> Self {
        Self::Range {
            start: x,
            end: Some(x),
        }
    }
}

impl From<Range<u64>> for SupportSet {
    fn from(range: Range<u64>) -> Self {
        if range.end <= range.start {
            return Self::empty();
        }
        Self::Range {
            start: range.start,
            end: Some(range.end - 1),
        }
    }
}

impl From<RangeInclusive<u64>> for SupportSet {
    fn from(range: RangeInclusive<u64>) -> Self {
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

impl From<RangeFrom<u64>> for SupportSet {
    fn from(range: RangeFrom<u64>) -> Self {
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
        }
    }
}

impl std::ops::AddAssign for SupportSet {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Mul<u64> for SupportSet {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        match self {
            Self::Empty => Self::Empty,
            Self::Range { start, end } => Self::Range {
                start: start * rhs,
                end: end.map(|x| x * rhs),
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
        }
    }
}

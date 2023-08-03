use std::ops::{Range, RangeFrom, RangeInclusive};

use num_traits::Zero;

use crate::number::{IntervalNumber, Rational};

/// Support set of a random variable (overapproximated as a range)
#[derive(Clone, Debug)]
pub enum SupportSet {
    Empty,
    Range { start: u32, end: Option<u32> },
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

    pub fn saturating_sub(self, other: u32) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Range { start, end } => Self::Range {
                start: start.saturating_sub(other),
                end: end.map(|x| x.saturating_sub(other)),
            },
            Self::Interval { start, end } => Self::Interval {
                start: (start - other.into()).max(&Rational::zero()),
                end: (end - other.into()).max(&Rational::zero()),
            },
        }
    }

    pub fn finite_range(&self) -> Option<std::ops::RangeInclusive<u32>> {
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
                start: start + start2,
                end: match (end, end2) {
                    (Some(x), Some(y)) => Some(x + y),
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

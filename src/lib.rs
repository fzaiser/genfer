#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_inception)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::single_match_else)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::bool_to_int_with_if)]
// TODO: get rid of as many of those `#![allow]`s as possible

pub mod bound;
pub mod interval;
pub mod numbers;
pub mod parser;
pub mod ppl;
pub mod semantics;
pub mod solvers;
pub mod support;
pub mod sym_expr;
pub mod util;

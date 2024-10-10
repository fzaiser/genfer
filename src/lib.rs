#![warn(clippy::pedantic)]
#![expect(clippy::module_name_repetitions)]
#![expect(clippy::must_use_candidate)]
#![expect(clippy::return_self_not_must_use)]
#![expect(clippy::cast_precision_loss)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_possible_wrap)]
#![expect(clippy::cast_sign_loss)]
#![expect(clippy::missing_panics_doc)]
#![expect(clippy::missing_errors_doc)]
#![expect(clippy::needless_pass_by_value)]
#![expect(clippy::comparison_chain)]
#![expect(clippy::assigning_clones)]

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

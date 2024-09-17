pub mod geometric;
pub mod residual;
pub mod support;

use crate::ppl::{Event, Program, Statement};

pub trait Transformer {
    type Domain;
    fn init(&mut self, program: &Program) -> Self::Domain;
    fn transform_event(
        &mut self,
        event: &Event,
        init: Self::Domain,
    ) -> (Self::Domain, Self::Domain);
    fn transform_statement(&mut self, stmt: &Statement, init: Self::Domain) -> Self::Domain;
    fn transform_statements(&mut self, stmts: &[Statement], init: Self::Domain) -> Self::Domain {
        let mut current = init;
        for stmt in stmts {
            current = self.transform_statement(stmt, current);
        }
        current
    }
    fn semantics(&mut self, program: &Program) -> Self::Domain {
        let init = self.init(program);
        self.transform_statements(&program.stmts, init)
    }
}

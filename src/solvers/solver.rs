use std::time::Duration;

use rustc_hash::FxHashMap;

use crate::{
    numbers::{FloatNumber, Rational},
    sym_expr::{SymConstraint, SymExpr, SymExprKind},
    util::{rational_to_z3, z3_real_to_rational},
};

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
    Other,
}

#[derive(Clone, Debug)]
pub struct ConstraintProblem {
    pub var_count: usize,
    pub decay_vars: Vec<usize>,
    pub factor_vars: Vec<usize>,
    pub block_vars: Vec<usize>,
    pub var_bounds: Vec<(Rational, Rational)>,
    pub constraints: Vec<SymConstraint>,
    /// Optimization objective (zero means no optimization)
    pub objective: SymExpr,
}

impl ConstraintProblem {
    pub fn holds_exact_f64(&self, assignment: &[f64]) -> bool {
        let assignment = assignment
            .iter()
            .map(|r| Rational::from(*r))
            .collect::<Vec<_>>();
        self.holds_exact(&assignment)
    }

    pub fn holds_exact(&self, assignment: &[Rational]) -> bool {
        self.holds_exact_with(assignment, &mut FxHashMap::default())
    }

    pub fn holds_exact_with(
        &self,
        assignment: &[Rational],
        cache: &mut FxHashMap<usize, Rational>,
    ) -> bool {
        self.all_constraints()
            .all(|c| c.holds_exact(assignment, cache))
    }

    pub fn objective_if_holds_exactly(&self, assignment: &[Rational]) -> Option<Rational> {
        let cache = &mut FxHashMap::default();
        if self.holds_exact_with(assignment, cache) {
            Some(self.objective.eval_exact(assignment, cache))
        } else {
            None
        }
    }

    pub fn all_constraints(&self) -> impl Iterator<Item = SymConstraint> + '_ {
        self.var_bounds
            .iter()
            .enumerate()
            .flat_map(move |(var, (lo, hi))| {
                let first = SymExpr::var(var).must_ge(lo.clone().into());
                if hi.is_finite() {
                    let second = SymExpr::var(var).must_lt(hi.clone().into());
                    vec![first, second]
                } else {
                    vec![first]
                }
            })
            .chain(self.constraints.iter().cloned())
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> Self {
        let cache = &mut FxHashMap::default();
        let objective = self.objective.substitute_with(replacements, cache);
        let constraints = self
            .constraints
            .iter()
            .map(|c| c.substitute_with(replacements, cache))
            .collect::<Vec<_>>();
        let var_count = self.var_count;
        let decay_vars = self.decay_vars.clone();
        let factor_vars = self.factor_vars.clone();
        let block_vars = self.block_vars.clone();
        let var_bounds = self.var_bounds.clone();
        Self {
            var_count,
            decay_vars,
            factor_vars,
            block_vars,
            var_bounds,
            constraints,
            objective,
        }
    }

    /// Detect and remove `<=` cycles between variables (arising from nested loops)
    ///
    /// Many numerical solvers have trouble with such cycles, so we remove them.
    /// Instead we replace the variables that have to be equal by one representative variable.
    pub fn preprocess(&mut self) {
        // Create a graph of `<=`-relations of variables
        let mut edges = vec![vec![]; self.var_count];
        for constraint in &self.constraints {
            match constraint {
                SymConstraint::Eq(a, b) => {
                    if let (SymExprKind::Variable(a), SymExprKind::Variable(b)) =
                        (a.kind(), b.kind())
                    {
                        edges[*a].push(*b);
                        edges[*b].push(*a);
                    }
                }
                SymConstraint::Le(a, b) => {
                    if let (SymExprKind::Variable(a), SymExprKind::Variable(b)) =
                        (a.kind(), b.kind())
                    {
                        edges[*a].push(*b);
                    }
                }
                _ => {}
            }
        }
        // The strongly-connected components are sets of variables that must be equal.
        // So for each SCC, we replace each of its variables by a canonical one.
        let sccs = Tarjan::new(self.var_count).sccs(&edges);
        let mut substitution = (0..self.var_count).map(SymExpr::var).collect::<Vec<_>>();
        for scc in &sccs {
            let canonical_var = scc.iter().min().unwrap();
            for var in scc {
                substitution[*var] = SymExpr::var(*canonical_var);
            }
        }
        *self = self.substitute(&substitution);
    }
}

/// Find strongly connected components using Tarjan's algorithm
struct Tarjan {
    index: usize,
    stack: Vec<usize>,
    scc: Vec<usize>,
    lowlink: Vec<usize>,
    on_stack: Vec<bool>,
    components: Vec<Vec<usize>>,
}

impl Tarjan {
    pub fn new(var_count: usize) -> Self {
        Self {
            index: 1,
            stack: Vec::new(),
            scc: vec![0; var_count],
            lowlink: vec![0; var_count],
            on_stack: vec![false; var_count],
            components: Vec::new(),
        }
    }

    pub fn sccs(mut self, edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
        // Nonrecursive Tarjan:
        for v in 0..edges.len() {
            if self.lowlink[v] == 0 {
                self.strongconnect(v, edges);
            }
        }
        self.components
    }

    fn strongconnect(&mut self, v: usize, edges: &[Vec<usize>]) {
        self.lowlink[v] = self.index;
        self.index += 1;
        self.scc[v] = self.lowlink[v];
        self.stack.push(v);
        self.on_stack[v] = true;
        for &w in &edges[v] {
            if self.lowlink[w] == 0 {
                self.strongconnect(w, edges);
                self.scc[v] = self.scc[v].min(self.scc[w]);
            } else if self.on_stack[w] {
                self.scc[v] = self.scc[v].min(self.lowlink[w]);
            }
        }
        if self.scc[v] == self.lowlink[v] {
            let mut component = Vec::new();
            loop {
                let w = self.stack.pop().unwrap();
                self.on_stack[w] = false;
                component.push(w);
                if w == v {
                    break;
                }
            }
            self.components.push(component);
        }
    }
}

pub trait Solver {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError>;
}

pub struct Z3Solver;

impl Solver for Z3Solver {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError> {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(timeout.as_millis() as u64);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);
        for constraint in problem.all_constraints() {
            solver.assert(&constraint.to_z3(&ctx, &rational_to_z3));
        }
        solver.push();
        match solver.check() {
            z3::SatResult::Unknown => {
                if let Some(reason) = solver.get_reason_unknown() {
                    if reason == "timeout" {
                        return Err(SolverError::Timeout);
                    }
                    panic!("Solver responded 'unknown': {reason}")
                } else {
                    panic!("Solver responded 'unknown' but no reason was given.")
                }
            }
            z3::SatResult::Unsat => return Err(SolverError::Infeasible),
            z3::SatResult::Sat => {}
        }
        let assignment = if let Some(model) = solver.get_model() {
            let mut assignment = Vec::new();
            for var in 0..problem.var_count {
                let val = model
                    .eval(&z3::ast::Real::new_const(&ctx, var as u32), false)
                    .unwrap();
                let val = z3_real_to_rational(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to rational"));
                assignment.push(val);
            }
            assignment
        } else {
            panic!("Solver returned SAT but no model.")
        };
        Ok(assignment)
    }
}

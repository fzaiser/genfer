use rustc_hash::FxHashMap;

use crate::{
    numbers::{FloatNumber, Rational},
    sym_expr::{SymConstraint, SymExpr, SymExprKind},
    util::rational_to_qepcad,
};

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
    pub(crate) fn holds_exact_f64(&self, assignment: &[f64]) -> bool {
        let assignment = assignment
            .iter()
            .map(|r| Rational::from(*r))
            .collect::<Vec<_>>();
        self.holds_exact(&assignment)
    }

    pub(crate) fn holds_exact(&self, assignment: &[Rational]) -> bool {
        self.holds_exact_with(assignment, &mut FxHashMap::default())
    }

    pub(crate) fn holds_exact_with(
        &self,
        assignment: &[Rational],
        cache: &mut FxHashMap<usize, Rational>,
    ) -> bool {
        let var_bounds_hold = assignment
            .iter()
            .zip(&self.var_bounds)
            .all(|(val, (lo, hi))| lo <= val && val < hi);
        let constraints_hold = self
            .constraints
            .iter()
            .all(|c| c.holds_exact(assignment, cache));
        var_bounds_hold && constraints_hold
    }

    pub fn objective_if_holds_exactly(&self, assignment: &[Rational]) -> Option<Rational> {
        let cache = &mut FxHashMap::default();
        if self.holds_exact_with(assignment, cache) {
            Some(self.objective.eval_exact(assignment, cache))
        } else {
            None
        }
    }

    pub(crate) fn substitute(&self, replacements: &[SymExpr]) -> Self {
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

    pub fn output_smtlib<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        writeln!(out, "(set-logic QF_NRA)")?;
        writeln!(out, "; for decimal output in Z3 (more readable):")?;
        writeln!(out, "(set-option :pp.decimal true)")?;
        writeln!(out)?;
        for i in 0..self.var_count {
            writeln!(out, "(declare-const {} Real)", SymExpr::var(i))?;
        }
        writeln!(out)?;
        for (v, (lo, hi)) in self.var_bounds.iter().enumerate() {
            writeln!(out, "(assert (<= {} {}))", lo, SymExpr::var(v))?;
            if hi.is_finite() {
                writeln!(out, "(assert (< {} {}))", SymExpr::var(v), hi)?;
            }
        }
        for constraint in &self.constraints {
            writeln!(out, "(assert {})", constraint.to_smtlib())?;
        }
        writeln!(out)?;
        writeln!(out, "(check-sat)")?;
        writeln!(out, "(get-model)")?;
        Ok(())
    }

    pub fn output_qepcad<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        // Name:
        writeln!(out, "[To run QEPCAD on this: $ cat thisfile | qepcad]")?;

        // List of variables:
        write!(out, "(")?;
        let mut first = true;
        for i in 0..self.var_count {
            if first {
                first = false;
            } else {
                write!(out, ", ")?;
            }
            write!(out, "{}", SymExpr::var(i))?;
        }
        writeln!(out, ")")?;

        // Number of free variables:
        writeln!(out, "2")?; // two variables for plotting

        // Formula:
        for i in 2..self.var_count {
            writeln!(out, "(E {})", SymExpr::var(i))?;
        }
        writeln!(out, "[")?;
        let mut first = true;
        for (v, (lo, hi)) in self.var_bounds.iter().enumerate() {
            if first {
                first = false;
            } else {
                writeln!(out, r" /\")?;
            }
            write!(out, "  {lo} <= {v}", v = SymExpr::var(v))?;
            if hi.is_finite() {
                write!(out, r" /\ {v} < {hi}", v = SymExpr::var(v))?;
            }
        }
        for c in &self.constraints {
            if first {
                first = false;
            } else {
                writeln!(out, r" /\")?;
            }
            write!(out, "  {}", c.to_qepcad(&rational_to_qepcad))?;
        }
        writeln!(out, "\n].")?;

        // Commands for various solving stages:
        writeln!(out, "go")?;
        writeln!(out, "go")?;
        writeln!(out, "go")?;
        writeln!(out, "p-2d-cad 0 1 0 1 0.0001 plot.eps")?; // 2D plot
        writeln!(out, "go")?;
        Ok(())
    }

    /// Detect and remove `<=` cycles between variables (arising from nested loops)
    ///
    /// Many numerical solvers have trouble with such cycles, so we remove them.
    /// Instead we replace the variables that have to be equal by one representative variable.
    pub fn preprocess(&mut self) {
        // Create a graph of `<=`-relations of variables
        let mut edges = vec![vec![]; self.var_count];
        for constraint in &self.constraints {
            if let (SymExprKind::Variable(a), SymExprKind::Variable(b)) =
                (constraint.lhs.kind(), constraint.rhs.kind())
            {
                edges[*a].push(*b);
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
    pub(crate) fn new(var_count: usize) -> Self {
        Self {
            index: 1,
            stack: Vec::new(),
            scc: vec![0; var_count],
            lowlink: vec![0; var_count],
            on_stack: vec![false; var_count],
            components: Vec::new(),
        }
    }

    pub(crate) fn sccs(mut self, edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
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

#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::float_cmp)]

use std::fmt::Write;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use genfer::parser::parse_program;
use genfer::ppl::{Comparison, Distribution, Event, Natural, Program, Statement, Var};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// The target language
    #[arg(value_enum)]
    target: Target,
    /// The file containing the probabilistic program
    file_name: PathBuf,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Target {
    Webppl,
    Anglican,
}

pub fn main() {
    let args = CliArgs::parse();
    let path = args.file_name;
    let program_name = path.file_stem().unwrap().to_str().unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    let program = parse_program(&contents);
    let mut output = String::new();
    match args.target {
        Target::Webppl => WebPpl::new(&mut output)
            .fmt_program(&program, program_name)
            .unwrap(),
        Target::Anglican => Anglican::new(&mut output)
            .fmt_program(&program, program_name)
            .unwrap(),
    }
    println!("{output}");
}

struct WebPpl<'a> {
    f: &'a mut dyn Write,
}

impl<'a> WebPpl<'a> {
    pub fn new(f: &'a mut dyn Write) -> Self {
        Self { f }
    }
}

trait Translate {
    fn fmt_program(&mut self, program: &Program, name: &str) -> std::fmt::Result;
    fn fmt_block(&mut self, stmts: &[Statement], indent: usize) -> std::fmt::Result;
    fn fmt_statement(&mut self, stmt: &Statement, indent: usize) -> std::fmt::Result;
    fn fmt_distribution(&mut self, dist: &Distribution) -> std::fmt::Result;
    fn fmt_event(&mut self, event: &Event) -> std::fmt::Result;
}

impl Translate for WebPpl<'_> {
    fn fmt_program(&mut self, program: &Program, name: &str) -> std::fmt::Result {
        writeln!(self.f, "var {name} = function() {{")?;
        for v in 0..program.used_vars().num_vars() {
            writeln!(self.f, "  {} = 0;", WebPplVar(Var(v)))?;
        }
        match program.stmts.as_slice() {
            [Statement::Normalize { given_vars, stmts }] if given_vars.is_empty() => {
                self.fmt_block(stmts, 2)?;
            }
            _ => {
                self.fmt_block(&program.stmts, 2)?;
            }
        }
        writeln!(self.f, "  return {};", WebPplVar(program.result))?;
        writeln!(self.f, "}};")?;
        writeln!(self.f, "var result = Infer({{ model: {name} }});")?;
        writeln!(self.f, "viz(result)")?;
        writeln!(self.f, "viz.table(result)")
    }

    fn fmt_block(&mut self, stmts: &[Statement], indent: usize) -> std::fmt::Result {
        for stmt in stmts {
            let indent_str = " ".repeat(indent);
            write!(self.f, "{indent_str}")?;
            self.fmt_statement(stmt, indent)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn fmt_statement(&mut self, stmt: &Statement, indent: usize) -> std::fmt::Result {
        use Statement::*;
        match stmt {
            Sample {
                var,
                distribution: dist,
                add_previous_value: add_previous,
            } => {
                let var = WebPplVar(*var);
                if *add_previous {
                    write!(self.f, "{var} += sample(")?;
                } else {
                    write!(self.f, "{var} = sample(")?;
                }
                self.fmt_distribution(dist)?;
                writeln!(self.f, ");")
            }
            Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                let var = WebPplVar(*var);
                if *add_previous_value {
                    write!(self.f, "{var} += ")?;
                } else {
                    write!(self.f, "{var} = ")?;
                }
                if let Some((coeff, var)) = addend {
                    let var = WebPplVar(*var);
                    if *coeff != Natural(1) {
                        write!(self.f, "{coeff} * ")?;
                    }
                    write!(self.f, "{var}")?;
                    if offset != &Natural(0) {
                        write!(self.f, " + {offset}")?;
                    }
                } else {
                    write!(self.f, "{offset}")?;
                }
                writeln!(self.f, ";")
            }
            Decrement { var, offset } => {
                let var = WebPplVar(*var);
                writeln!(
                    self.f,
                    "{var} = ({var} < {offset}) ? 0 : ({var} - {offset});"
                )
            }
            IfThenElse { cond, then, els } => {
                if let Some(event) = stmt.recognize_observe() {
                    if let Event::DataFromDist(data, dist) = event {
                        write!(self.f, "observe(")?;
                        self.fmt_distribution(dist)?;
                        writeln!(self.f, ", {data});")?;
                    } else {
                        write!(self.f, "condition(")?;
                        self.fmt_event(event)?;
                        writeln!(self.f, ");")?;
                    }
                    return Ok(());
                }
                write!(self.f, "if (")?;
                self.fmt_event(cond)?;
                writeln!(self.f, ") {{")?;
                self.fmt_block(then, indent + 2)?;
                let indent_str = " ".repeat(indent);
                match els.as_slice() {
                    [] => writeln!(self.f, "{indent_str}}}")?,
                    [if_stmt @ IfThenElse { .. }] if if_stmt.recognize_observe().is_none() => {
                        write!(self.f, "{indent_str}}} else ")?;
                        self.fmt_statement(if_stmt, indent)?;
                    }
                    _ => {
                        writeln!(self.f, "{indent_str}}} else {{")?;
                        self.fmt_block(els, indent + 2)?;
                        writeln!(self.f, "{indent_str}}}")?;
                    }
                }
                Ok(())
            }
            Fail => writeln!(self.f, "condition(false);"),
            Normalize { given_vars, stmts } => {
                let indent_str = " ".repeat(indent);
                let num_vars = stmt.used_vars().num_vars();
                for v in 0..num_vars {
                    if given_vars.contains(&Var(v)) {
                        continue;
                    }
                    let webppl_v = WebPplVar(Var(v));
                    writeln!(self.f, "if ({webppl_v} != 0) {{ error('This form of nested inference is not supported in WebPPL: the variable `{webppl_v}` should either be unassigned (i.e. 0) at this point or part of the `normalize` statement.'); }}")?;
                    write!(self.f, "{indent_str}")?;
                }
                writeln!(self.f, "var assignment = sample(Infer(function(){{")?;
                self.fmt_block(stmts, indent + 2)?;
                let vars = (0..num_vars).map(|v| WebPplVar(Var(v)));
                writeln!(
                    self.f,
                    "{indent_str}  return [{}];",
                    vars.map(|v| format!("{v}, ")).collect::<String>()
                )?;
                writeln!(self.f, "{indent_str}}}));")?;
                for v in 0..num_vars {
                    let webppl_v = WebPplVar(Var(v));
                    writeln!(self.f, "{indent_str}{webppl_v} = assignment[{v}];")?;
                }
                Ok(())
            }
        }
    }

    fn fmt_distribution(&mut self, dist: &Distribution) -> std::fmt::Result {
        use Distribution::*;
        match dist {
            Dirac(a) => write!(self.f, "Delta({{v: {a}}}"),
            // WebPPL's Bernoulli(p) distribution yields `false` and `true`, not 0 and 1, so we use Binomial(1, p) instead.
            Bernoulli(p) => write!(self.f, "Binomial({{n: 1, p: {p}}})"),
            BernoulliVarProb(v) => write!(self.f, "Binomial({{n: 1, p: {v}}})", v = WebPplVar(*v)),
            BinomialVarTrials(n, p) => write!(
                self.f,
                "({n} == 0 ? Delta({{v: 0}}) : Binomial({{n: {n}, p: {p}}}))",
                n = WebPplVar(*n)
            ),
            Binomial(n, p) => {
                if n == &Natural(0) {
                    write!(self.f, "Delta({{v: 0}})")
                } else {
                    write!(self.f, "Binomial({{n: {n}, p: {p}}})")
                }
            }
            Categorical(rs) => {
                write!(self.f, "Categorical({{ ps: [")?;
                for i in 0..rs.len() {
                    write!(self.f, "{i}, ")?;
                }
                write!(self.f, "], vs: [")?;
                for r in rs {
                    write!(self.f, "{r}, ")?;
                }
                write!(self.f, "] }})")
            }
            NegBinomialVarSuccesses(_, _) | NegBinomial(_, _) => {
                panic!("Negative binomial distribution is not supported by WebPPL")
            }
            Geometric(p) => {
                // Since WebPPL does not have a Geometric distribution, we approximate it with a Categorical distribution:
                let threshold = 1e-6;
                let (vs, ps): (String, String) = (0..100)
                    .map(|i| (i, p.round() * p.complement().round().powi(i)))
                    .take_while(|(_, p)| p.round() > threshold)
                    .map(|(i, p)| (format!("{i}, "), format!("{p}, ")))
                    .unzip();
                write!(self.f, "Categorical({{ ps: [{ps}], vs: [{vs}] }})")
            }
            Poisson(lambda) => {
                if lambda.is_zero() {
                    write!(self.f, "Delta({{v: 0}})")
                } else {
                    write!(self.f, "Poisson({{mu: {lambda}}})")
                }
            }
            PoissonVarRate(lambda, mu) => write!(
                self.f,
                "({lambda} * {mu} == 0 ? Delta({{v: 0}}) : Poisson({{mu: {lambda} * {mu}}}))",
                mu = WebPplVar(*mu)
            ),
            Uniform { start, end } => {
                if start == &Natural(0) {
                    write!(self.f, "RandomInteger({{n: {}}})", *end)
                } else {
                    panic!("Uniform distribution is not supported by WebPPL")
                }
            }
            Exponential { rate } => write!(self.f, "Exponential({{a: {rate}}})"),
            Gamma { shape, rate } => {
                write!(
                    self.f,
                    "Gamma({{shape: {shape}, scale: {}}})",
                    1.0 / rate.round()
                )
            }
            UniformCont { start, end } => {
                write!(self.f, "Uniform({{a: {start}, b: {end}}})")
            }
        }
    }

    fn fmt_event(&mut self, event: &Event) -> std::fmt::Result {
        match event {
            Event::InSet(v, set) => {
                let var = WebPplVar(*v);
                let mut first = true;
                for i in set {
                    if first {
                        first = false;
                    } else {
                        write!(self.f, " || ")?;
                    }
                    write!(self.f, "{var} === {i}")?;
                }
                Ok(())
            }
            Event::VarComparison(lhs, comp, rhs) => {
                let lhs = WebPplVar(*lhs);
                let rhs = WebPplVar(*rhs);
                match comp {
                    Comparison::Eq => write!(self.f, "{lhs} === {rhs}"),
                    Comparison::Lt => write!(self.f, "{lhs} < {rhs}"),
                    Comparison::Le => write!(self.f, "{lhs} <= {rhs}"),
                }
            }
            Event::DataFromDist(data, dist) => {
                write!(self.f, "sample(")?;
                self.fmt_distribution(dist)?;
                write!(self.f, ") === {data}")
            }
            Event::Complement(e) => {
                write!(self.f, "!(")?;
                self.fmt_event(e)?;
                write!(self.f, ")")
            }
            Event::Intersection(es) => {
                let mut first = true;
                for e in es {
                    if first {
                        first = false;
                        write!(self.f, "(")?;
                    } else {
                        write!(self.f, " && ")?;
                    }
                    self.fmt_event(e)?;
                }
                write!(self.f, ")")
            }
        }
    }
}

#[derive(Clone, Debug)]
struct WebPplVar(Var);

impl std::fmt::Display for WebPplVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "globalStore.{}", self.0)
    }
}

struct Anglican<'a> {
    f: &'a mut dyn Write,
    num_vars: usize,
    nested: Vec<String>,
}

impl<'a> Anglican<'a> {
    pub fn new(f: &'a mut dyn Write) -> Self {
        Self {
            f,
            num_vars: 0,
            nested: Vec::new(),
        }
    }

    fn var_list(&mut self) -> String {
        let mut output = String::new();
        let num_vars = self.num_vars;
        for var in 0..num_vars {
            let var = Var(var);
            write!(&mut output, " {var}").unwrap();
        }
        output
    }
}

impl<'a> Translate for Anglican<'a> {
    #[allow(clippy::too_many_lines)]
    fn fmt_program(&mut self, program: &Program, name: &str) -> std::fmt::Result {
        self.num_vars = program.used_vars().num_vars();
        let var_list = self.var_list();
        writeln!(
            self.f,
            r#"
(ns model
  (:require [gorilla-plot.core :as plot])
  (:use [anglican core emit runtime stat
          [state :only [get-predicts get-log-weight get-result]]]))

(defdist geometric
"Geometric distribution on support {{0,1,2....}}"
[p] []
(sample* [this]
        (loop [value 0]
            (if (sample* (flip p))
            value
            (recur (inc value)))))
(observe* [this value] (+ (log p) (* value (log (- 1 p))))))

(defdist dirac [x]
    (sample* [this] x)
    (observe* [this value]
              (if (= value x)
                0
                (- (/ 1.0 0.0)))))

"#
        )?;
        writeln!(self.f, "(with-primitive-procedures [dirac geometric]")?;
        let mut main_query = String::new();
        let mut new_self = Anglican {
            f: &mut main_query,
            num_vars: self.num_vars,
            nested: Vec::new(),
        };
        writeln!(new_self.f, "  (defquery model [method- options- ]")?;
        writeln!(
            new_self.f,
            "    (let [[{var_list}] [ {}]",
            "0 ".repeat(self.num_vars)
        )?;
        writeln!(new_self.f, "          [{var_list}]")?;
        match program.stmts.as_slice() {
            [Statement::Normalize { given_vars, stmts }] if given_vars.is_empty() => {
                new_self.fmt_block(stmts, 10)?;
            }
            _ => new_self.fmt_block(&program.stmts, 10)?,
        }
        writeln!(new_self.f, "         ]")?;
        writeln!(new_self.f, "    {}", program.result)?;
        writeln!(new_self.f, "    )")?;
        writeln!(new_self.f, "  )")?;
        self.nested = new_self.nested;
        // Write nested queries in reverse order to ensure they are defined before use
        for (i, nested) in self.nested.iter().enumerate().rev() {
            writeln!(
                self.f,
                "  (defquery nested{i} [method- options- {var_list}]"
            )?;
            writeln!(self.f, "{nested}")?;
            writeln!(self.f, "  )")?;
        }
        writeln!(self.f, "{main_query}")?;
        writeln!(self.f, ")\n\n")?;
        writeln!(self.f, r#"(def model_name "{name}")"#)?;
        writeln!(self.f, r#"(def outfile "{name}_anglican.json")"#)?;
        #[allow(clippy::write_literal)] // to avoid escaping "{}" in the string
        writeln!(
            self.f,
            "{}",
            r#"
; (def configurations [:rmh []])
(def configurations
  [
    [:importance []]
    [:lmh []]
    [:rmh []]
    [:smc []]
    [:smc [:number-of-particles 100]]
    [:pgibbs []]
    [:ipmcmc []]
  ])

; (def num_samples_options [1000])
(def num_samples_options [1000 10000])
(def thinning 1)

(spit outfile "[\n" :append false)

(def num-chains 20)

(doall
  (for [ num_samples num_samples_options
         [method options] configurations
         chain (range 0 num-chains)]
    (do
      (println (format "\nMethod %s with %s samples and options %s" method num_samples options))
      (println (format "Chain no. %s" chain))
      (let [start (. System (nanoTime))
            warmup (/ num_samples 5)
            samples (take-nth thinning (take (* num_samples thinning) (drop warmup (apply doquery method model [method options] options))))
            results (collect-results samples)
            values (map (fn [s] (get-result s)) samples)
            max-value (apply max values)
            mean (empirical-mean results)
            variance (empirical-variance results)
            std (empirical-std results)
            skewness (if (zero? std) (/ 0.0 0.0) (empirical-skew results))
            kurtosis (if (zero? std) (/ 0.0 0.0) (empirical-kurtosis results))
            distribution (empirical-distribution (collect-results samples))
            masses (for [n (range 0 (inc max-value))] (get distribution n 0.0))
            end (. System (nanoTime))
            elapsed_ms (/ (- end start) 1e6)]
        (println (format "Elapsed time: %s ms" elapsed_ms))
        (println (format "Empirical mean: %s" mean))
        (println (format "Empirical variance: %s" variance))
        (println (format "Empirical std: %s" std))
        (println (format "Empirical skewness: %s" skewness))
        (println (format "Empirical kurtosis: %s" kurtosis))
        (spit outfile (format
                   "{\"model\": \"%s\", \"system\": \"anglican\", \"method\": \"%s\", \"options\": \"%s\", \"num_samples\": %s, \"time_ms\": %s, \"total\": 1.0, \"mean\": %s, \"variance\": %s, \"stddev\": %s, \"skewness\": %s, \"kurtosis\": %s, \"masses\": [%s] },\n"
                   model_name method options num_samples elapsed_ms mean variance std skewness kurtosis
                   (clojure.string/join ", " masses)) :append true)
        (if false (do
          (println "Empirical distribution:")
          (doall (for [n (range 0 (inc max-value))]
            (println (format "p(%s) = %s" n (get distribution n 0.0)))))))
        ; (println "List of samples (format: sample log-weight):")
        ; (doall (map (fn [s] (println (format "%s %s" (get-result s) (get-log-weight s)))) samples))
        ; values need to be adjusted if they are weighted!
        ; (plot/histogram values :normalize :probability)
      )
    )
  )
)

(spit outfile "]\n" :append true)

"#
        )?;
        Ok(())
    }

    fn fmt_block(&mut self, stmts: &[Statement], indent: usize) -> std::fmt::Result {
        let var_list = self.var_list();
        let indent_str = " ".repeat(indent);
        if stmts.is_empty() {
            writeln!(self.f, "{indent_str}[{var_list} ]")?;
            return Ok(());
        }
        writeln!(self.f, "{indent_str}(let [")?;
        let var_indent_str = " ".repeat(indent + 6);
        for stmt in stmts {
            write!(self.f, "{var_indent_str}")?;
            self.fmt_statement(stmt, indent + 6)?;
            writeln!(self.f)?;
        }
        writeln!(self.f, "{indent_str}     ]")?;
        writeln!(self.f, "{indent_str}  [{var_list} ]")?;
        writeln!(self.f, "{indent_str})")?;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn fmt_statement(&mut self, stmt: &Statement, indent: usize) -> std::fmt::Result {
        use Statement::*;
        let indent_str = " ".repeat(indent);
        match stmt {
            Sample {
                var,
                distribution,
                add_previous_value,
            } => {
                write!(self.f, "{var} ")?;
                if *add_previous_value {
                    write!(self.f, "(+ {var} (sample ")?;
                    self.fmt_distribution(distribution)?;
                    write!(self.f, "))")?;
                } else {
                    write!(self.f, "(sample ")?;
                    self.fmt_distribution(distribution)?;
                    write!(self.f, ")")?;
                }
                Ok(())
            }
            Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                write!(self.f, "{var} (+")?;
                if *add_previous_value {
                    write!(self.f, " {var}")?;
                }
                if let Some((factor, var)) = addend {
                    write!(self.f, " (* {factor} {var})")?;
                } else {
                    write!(self.f, " 0")?;
                }
                write!(self.f, " {offset})")?;
                Ok(())
            }
            Decrement { var, offset } => {
                write!(self.f, "{var} (if (< {var} {offset}) 0 (- {var} {offset}))")
            }
            IfThenElse { els, .. } => {
                if let Some(event) = stmt.recognize_observe() {
                    write!(self.f, "_unused ")?;
                    if let Event::DataFromDist(data, dist) = event {
                        write!(self.f, "(observe ")?;
                        self.fmt_distribution(dist)?;
                        write!(self.f, " {data})")?;
                    } else {
                        write!(self.f, "(observe (flip 1.0) ")?;
                        self.fmt_event(event)?;
                        write!(self.f, ")")?;
                    }
                    return Ok(());
                }
                let var_list = self.var_list();
                writeln!(self.f, "[{var_list}] (cond")?;
                let mut statement = stmt;
                let mut rest = els;
                while let IfThenElse { cond, then, els } = statement {
                    write!(self.f, "{indent_str}  ")?;
                    self.fmt_event(cond)?;
                    writeln!(self.f)?;
                    self.fmt_block(then, indent + 2)?;
                    if let [ite @ IfThenElse { .. }] = els.as_slice() {
                        if ite.recognize_observe().is_none() {
                            statement = ite;
                            continue;
                        }
                    }
                    rest = els;
                    break;
                }
                writeln!(self.f, "{indent_str}  :else")?;
                self.fmt_block(rest, indent + 2)?;
                write!(self.f, "{indent_str})")
            }
            Fail => write!(self.f, "_ (observe (flip 1.0) false)"),
            Normalize { given_vars, stmts } => {
                for v in 0..self.num_vars {
                    let anglican_v = Var(v);
                    if !given_vars.contains(&anglican_v) {
                        write!(self.f, "_unused (assert (= {anglican_v} 0) \"This form of nested inference is not supported in Anglican: the variable `{anglican_v}` should either be unassigned (i.e. 0) at this point or part of the `normalize` statement.\")\n{indent_str}")?;
                    }
                }
                let var_list = self.var_list();
                let nested_id = self.nested.len();
                let mut output = String::new();
                let mut new_self = Anglican {
                    f: &mut output,
                    num_vars: self.num_vars,
                    nested: self.nested.clone(),
                };
                new_self.fmt_block(stmts, 4)?;
                let nested_outputs = new_self.nested;
                self.nested.push(output);
                self.nested
                    .extend(nested_outputs.into_iter().skip(nested_id + 1));
                writeln!(
                    self.f,
                    "[{var_list}] (sample ((apply conditional nested{nested_id} method- options-) method- options- {var_list}))",
                )?;
                Ok(())
            }
        }
    }

    fn fmt_distribution(&mut self, dist: &Distribution) -> std::fmt::Result {
        match dist {
            Distribution::Dirac(a) => write!(self.f, "(dirac {})", a.round()),
            Distribution::Bernoulli(p) => write!(self.f, "(bernoulli {})", p.round()),
            Distribution::BernoulliVarProb(v) => write!(self.f, "(bernoulli {v})"),
            Distribution::BinomialVarTrials(n, p) => write!(self.f, "(binomial {n} {})", p.round()),
            Distribution::Binomial(n, p) => write!(self.f, "(binomial {n} {})", p.round()),
            Distribution::Categorical(rs) => {
                write!(self.f, "(categorical [")?;
                for (i, r) in rs.iter().enumerate() {
                    write!(self.f, "[{i} {}] ", r.round())?;
                }
                write!(self.f, "])")
            }
            Distribution::NegBinomialVarSuccesses(_, _) | Distribution::NegBinomial(_, _) => {
                panic!("Negative binomial distribution is not supported by Anglican")
            }
            Distribution::Geometric(p) => write!(self.f, "(geometric {})", p.round()),
            Distribution::Poisson(lambda) => {
                if lambda.is_zero() {
                    write!(self.f, "(dirac 0)")
                } else {
                    write!(self.f, "(poisson {})", lambda.round())
                }
            }
            Distribution::PoissonVarRate(lambda, mu) => write!(
                self.f,
                "(if (zero? (* {lambda} {mu})) (dirac 0) (poisson (* {lambda} {mu})))",
                lambda = lambda.round(),
            ),
            Distribution::Uniform { start, end } => {
                write!(self.f, "(uniform-discrete {start} {})", *end)
            }
            Distribution::Exponential { rate } => write!(self.f, "(exponential {})", rate.round()),
            Distribution::Gamma { shape, rate } => {
                write!(self.f, "(gamma {} {})", shape.round(), rate.round())
            }
            Distribution::UniformCont { start, end } => {
                write!(
                    self.f,
                    "(uniform-continuous {} {})",
                    start.round(),
                    end.round()
                )
            }
        }
    }

    fn fmt_event(&mut self, event: &Event) -> std::fmt::Result {
        match event {
            Event::InSet(v, set) => {
                write!(self.f, "(contains? [ ")?;
                for i in set {
                    write!(self.f, "{i} ")?;
                }
                write!(self.f, "] {v})")
            }
            Event::VarComparison(lhs, comp, rhs) => match comp {
                Comparison::Eq => write!(self.f, "(= {lhs} {rhs})"),
                Comparison::Lt => write!(self.f, "(< {lhs} {rhs})"),
                Comparison::Le => write!(self.f, "(<= {lhs} {rhs})"),
            },
            Event::DataFromDist(data, dist) => {
                write!(self.f, "(= (sample ")?;
                self.fmt_distribution(dist)?;
                write!(self.f, ") {data})")
            }
            Event::Complement(e) => {
                write!(self.f, "(not ")?;
                self.fmt_event(e)?;
                write!(self.f, ")")
            }
            Event::Intersection(es) => {
                write!(self.f, "(and")?;
                for e in es {
                    write!(self.f, " ")?;
                    self.fmt_event(e)?;
                }
                write!(self.f, ")")
            }
        }
    }
}

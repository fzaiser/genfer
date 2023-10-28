#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::cmp::Ordering;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use genfer::generating_function::{
    central_to_standardized_moments, moments_taylor, moments_to_central_moments, probs_taylor,
    GenFun,
};
use genfer::interval::Interval;
use genfer::number::{
    BigFloat, FloatNumber, IntervalNumber, MultiPrecFloat, Number, Rational, F64, PRECISION,
};
use genfer::parser;
use genfer::ppl::{GfTranslation, Program};
use num_traits::{One, Zero};

use clap::Parser;
use genfer::support::SupportSet;
use genfer::symbolic::{moments_symbolic, probs_symbolic};

const MAX_PROB_LIMIT: usize = 1000;

#[allow(clippy::struct_excessive_bools)]
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// The file containing the probabilistic program
    file_name: PathBuf,
    /// Use floats with a wider exponent to prevent under-/overflow
    #[arg(long, group = "number representation")]
    big_float: bool,
    /// Use floating point numbers with the given number of bits of precision
    ///
    /// Arbitrary precision floating point numbers are provided by the MPFR library.
    /// This option does not guarantee that the result is correct to that precision,
    /// because rounding errors can still compound.
    /// But it can reduce the impact of rounding errors compared to the default precision of 53 bits.
    /// Together with the `--bounds` option, the actual precision of the result can be verified.
    #[arg(short, long, group = "number representation")]
    precision: Option<u32>,
    /// Use rational numbers instead of floating point numbers
    ///
    /// This avoids rounding errors but is only possible if the results are rational.
    #[arg(short, long, group = "number representation")]
    rational: bool,
    /// Bound the floating-point rounding errors using interval arithmetic
    ///
    /// With this option, the output is guaranteed to be correct, taking rounding errors into account.
    /// In order to reduce the rounding errors and tighten the bounds,
    /// the option `--precision` can be useful.
    #[arg(short, long)]
    bounds: bool,
    /// Whether to skip simplification of the generating function before evaluating it
    ///
    /// This tends to be most beneficial for programs with finite distributions.
    #[arg(long)]
    no_simplify_gf: bool,
    /// Represent generating functions symbolically (instead of Taylor series)
    #[arg(short, long)]
    symbolic: bool,
    /// Print the parsed probabilistic program
    #[arg(long)]
    print_program: bool,
    /// Print the generating function
    #[arg(long)]
    print_gf: bool,
    /// Disable timing of the execution
    #[arg(long)]
    no_timing: bool,
    /// Disable printing of probability masses
    #[arg(long)]
    no_probs: bool,
    /// The limit for the probability masses to be computed
    ///
    /// By default, it is determined automatically such that the remaining normalized probability mass is guaranteed
    /// to be < 1/256 by Markov's inequality, but capped at 1000.
    #[arg(short, long)]
    limit: Option<usize>,
    /// Write the results to a JSON file
    #[arg(long)]
    json: Option<PathBuf>,
}

pub fn main() {
    let args = CliArgs::parse();
    let contents = std::fs::read_to_string(&args.file_name).unwrap();
    let program = parser::parse_program(&contents);
    if args.print_program {
        println!("Parsed program:\n{program}\n");
    }
    if args.bounds {
        if args.rational {
            run_program_intervals::<Rational>(&program, &args);
        } else if let Some(prec) = args.precision {
            PRECISION.with(|p| {
                p.set(prec).unwrap();
            });
            run_program_intervals::<MultiPrecFloat>(&program, &args);
        } else if args.big_float {
            run_program_intervals::<BigFloat>(&program, &args);
        } else {
            run_program_intervals::<F64>(&program, &args);
        }
    } else {
        #[allow(clippy::collapsible_else_if)]
        if args.rational {
            run_program::<Rational>(&program, &args);
        } else if let Some(prec) = args.precision {
            PRECISION.with(|p| {
                p.set(prec).unwrap();
            });
            run_program::<MultiPrecFloat>(&program, &args);
        } else if args.big_float {
            run_program::<BigFloat>(&program, &args);
        } else {
            run_program::<F64>(&program, &args);
        }
    }
}

fn run_program_intervals<T: IntervalNumber + Into<f64>>(program: &Program, args: &CliArgs) {
    let inference_start = Instant::now();
    let uses_observe = program.uses_observe();
    let GfTranslation { var_info, gf } = translate_program_to_gf::<Interval<T>>(program, args);
    let gf_translation_time = inference_start.elapsed();
    if args.symbolic {
        let gf = gf.to_computation();
        print_moments_and_probs_interval(
            |limit| moments_symbolic(&gf, program.result, &var_info, limit),
            |limit| probs_symbolic(&gf, program.result, &var_info, limit),
            &var_info[program.result.id()],
            uses_observe,
            args,
            inference_start,
            gf_translation_time,
        );
    } else {
        print_moments_and_probs_interval(
            |limit| moments_taylor(&gf, program.result, &var_info, limit),
            |limit| probs_taylor(&gf, program.result, &var_info, limit),
            &var_info[program.result.id()],
            uses_observe,
            args,
            inference_start,
            gf_translation_time,
        );
    }
}

fn run_program<T: IntervalNumber + Into<f64>>(program: &Program, args: &CliArgs) {
    let inference_start = Instant::now();
    let uses_observe = program.uses_observe();
    let GfTranslation { var_info, gf } = translate_program_to_gf::<T>(program, args);
    let gf_translation_time = inference_start.elapsed();
    if args.symbolic {
        let gf = gf.to_computation();
        print_moments_and_probs(
            |limit| moments_symbolic(&gf, program.result, &var_info, limit),
            |limit| probs_symbolic(&gf, program.result, &var_info, limit),
            &var_info[program.result.id()],
            uses_observe,
            args,
            inference_start,
            gf_translation_time,
        );
    } else {
        print_moments_and_probs(
            |limit| moments_taylor(&gf, program.result, &var_info, limit),
            |limit| probs_taylor(&gf, program.result, &var_info, limit),
            &var_info[program.result.id()],
            uses_observe,
            args,
            inference_start,
            gf_translation_time,
        );
    }
}

fn translate_program_to_gf<T: Number>(program: &Program, args: &CliArgs) -> GfTranslation<T> {
    let start = Instant::now();
    let translation = program.transform_gf::<T>(GenFun::one());
    let translation = if args.no_simplify_gf {
        translation
    } else {
        let gf = translation.gf.simplify();
        let var_info = translation.var_info;
        GfTranslation { var_info, gf }
    };
    if args.print_gf {
        println!("Generating function:\n{}\n", translation.gf);
    }
    print_elapsed_message(start, "Time to construct the generating function: ", args);
    translation
}

fn print_moments_and_probs<T: IntervalNumber + Into<f64>>(
    moments_fn: impl Fn(usize) -> (T, Vec<T>),
    probs_fn: impl Fn(usize) -> Vec<T>,
    var_info: &SupportSet,
    uses_observe: bool,
    args: &CliArgs,
    inference_start: Instant,
    gf_translation_time: Duration,
) {
    print_moments_and_probs_interval(
        |limit| {
            let (total, moments) = moments_fn(limit);
            (
                Interval::precisely(total),
                moments.into_iter().map(Interval::precisely).collect(),
            )
        },
        |limit| {
            probs_fn(limit)
                .into_iter()
                .map(Interval::precisely)
                .collect()
        },
        var_info,
        uses_observe,
        args,
        inference_start,
        gf_translation_time,
    );
}

fn in_interval<T: IntervalNumber>(iv: &Interval<T>, print_intervals: bool) -> String {
    if let Some(x) = iv.extract_point() {
        format!("= {x}")
    } else if !print_intervals {
        format!("= {}", iv.clone().center())
    } else {
        format!("∈ [{}, {}]", iv.lo, iv.hi)
    }
}

fn print_moments_and_probs_interval<T: IntervalNumber + Into<f64>>(
    moments_fn: impl Fn(usize) -> (Interval<T>, Vec<Interval<T>>),
    probs_fn: impl Fn(usize) -> Vec<Interval<T>>,
    var_info: &SupportSet,
    uses_observe: bool,
    args: &CliArgs,
    inference_start: Instant,
    gf_translation_time: Duration,
) {
    println!("Support is a subset of: {var_info}");
    println!();
    println!("Computing moments...");
    let moment_start = Instant::now();
    let (total, moments) = moments_fn(5);
    let total = total
        .ensure_lower_bound(T::zero())
        .ensure_upper_bound(T::one());
    let moments = moments
        .into_iter()
        .map(|x| x.ensure_lower_bound(T::zero()))
        .collect::<Vec<_>>();
    let mut moments_struct = moments_to_moments_struct(total.clone(), &moments);
    moments_struct.variance = moments_struct.variance.ensure_lower_bound(T::zero());
    moments_struct.stddev = moments_struct.stddev.ensure_lower_bound(T::zero());
    moments_struct.kurtosis = moments_struct.kurtosis.ensure_lower_bound(T::zero());
    print_moments(&moments_struct, args.bounds);
    let time_for_moments = moment_start.elapsed();
    print_elapsed_message(moment_start, "Time to compute moments: ", args);
    let probs_data = if args.no_probs || !var_info.is_discrete() || total.is_zero() {
        None
    } else {
        let probs_start = Instant::now();
        let probs = print_probs(
            args,
            &total,
            &moments,
            var_info,
            uses_observe,
            probs_fn,
            probs_start,
        );
        Some((probs, probs_start.elapsed()))
    };
    print_elapsed_message(inference_start, "Total inference time: ", args);
    if let Some(json_path) = &args.json {
        let moment_data = (&moments_struct.map(Interval::center), time_for_moments);
        let probs_data = probs_data
            .map(|(ivs, t)| (ivs.into_iter().map(Interval::center).collect::<Vec<_>>(), t));
        print_json(
            moment_data,
            &probs_data,
            gf_translation_time,
            inference_start.elapsed(),
            args,
            json_path,
        )
        .expect("failed to write JSON file");
    }
}

fn print_probs<T: IntervalNumber + Into<f64>>(
    args: &CliArgs,
    total: &Interval<T>,
    moments: &[Interval<T>],
    var_info: &SupportSet,
    uses_observe: bool,
    probs_fn: impl Fn(usize) -> Vec<Interval<T>>,
    probs_start: Instant,
) -> Vec<Interval<T>> {
    println!();
    let limit = if let Some(limit) = args.limit {
        limit
    } else if total.is_zero() {
        1
    } else if let Some(range) = var_info.finite_range() {
        *range.end() as usize + 1
    } else {
        // Markov's inequality ensures that P(X >= limit) <= 1 / 4.0^4 = 1 / 256.
        // For practicality, we cap the limit at MAX_PROB_LIMIT.
        let (mean, central_moments) = moments_to_central_moments(moments);
        let central4th_root = central_moments[2].hi.clone().into().sqrt().sqrt();
        let limit = (mean.hi.into() + 4.0 * central4th_root).ceil();
        if limit.is_finite() {
            (limit as usize + 1).min(MAX_PROB_LIMIT)
        } else {
            println!("Failed to find a limit automatically due to non-finite moments.");
            println!("Please specify a limit manually with `--limit`.");
            println!("Using a limit of 2 for now.");
            2
        }
    };
    println!("Computing probabilities up to {limit}...");
    let is_normalized = !uses_observe || total.is_one();
    let mut mass_missing = total.clone();
    let mut probs = probs_fn(limit);
    let mut normalized_probs = Vec::new();
    for i in 0..limit {
        let p = probs[i].clone();
        assert!(
            !(p < Interval::zero() || p > Interval::one()),
            "p({i}) = {p} is not a probability"
        );
        let p = p.ensure_lower_bound(T::zero()).ensure_upper_bound(T::one());
        probs[i] = p.clone();
        if is_normalized {
            println!("p({i}) {}", in_interval(&p, args.bounds));
        } else {
            let unnormalized = in_interval(&p, args.bounds);
            let normalized_p = p.clone() / total.clone();
            let normalized_p = normalized_p
                .ensure_lower_bound(T::zero())
                .ensure_upper_bound(T::one());
            let normalized = in_interval(&normalized_p, args.bounds);
            println!("Unnormalized: p({i})     {unnormalized}");
            println!("Normalized:   p({i}) / Z {normalized}");
            normalized_probs.push(normalized_p);
        }
        mass_missing -= p.clone();
    }
    if let Some(range) = var_info.finite_range() {
        if (*range.end() as usize) < limit {
            mass_missing = Interval::zero();
        }
    }
    let mass_missing_unnorm = mass_missing.hi.max(&T::zero()).min(&T::one());
    let mass_missing_norm = (mass_missing / total.clone())
        .hi
        .max(&T::zero())
        .min(&T::one());
    if is_normalized {
        println!("p(n) <= {mass_missing_unnorm} for all n >= {limit}");
    } else {
        println!("Unnormalized: p(n)     <= {mass_missing_unnorm} for all n >= {limit}");
        println!("Normalized:   p(n) / Z <= {mass_missing_norm} for all n >= {limit}");
    }
    print_elapsed_message(probs_start, "Time to compute probability masses: ", args);
    normalized_probs
}

#[derive(Clone, Debug)]
struct Moments<T> {
    total: T,
    mean: T,
    variance: T,
    stddev: T,
    skewness: T,
    kurtosis: T,
}

impl<T> Moments<T> {
    fn map<U>(self, mut f: impl FnMut(T) -> U) -> Moments<U> {
        Moments {
            total: f(self.total),
            mean: f(self.mean),
            variance: f(self.variance),
            stddev: f(self.stddev),
            skewness: f(self.skewness),
            kurtosis: f(self.kurtosis),
        }
    }
}

fn moments_to_moments_struct<T: FloatNumber + PartialOrd>(total: T, moments: &[T]) -> Moments<T> {
    let (mean, central_moments) = moments_to_central_moments(moments);
    let (variance, std_moments) = central_to_standardized_moments(&central_moments);
    let skewness = std_moments[0].clone();
    let kurtosis = std_moments[1].clone();
    let stddev = variance.sqrt();
    assert!(
        moments
            .iter()
            .all(|x| x.partial_cmp(&T::zero()) != Some(Ordering::Less)),
        "moments must be non-negative for distributions supported on the natural numbers"
    );
    assert!(
        variance.partial_cmp(&T::zero()) != Some(Ordering::Less),
        "variance must be non-negative"
    );
    assert!(
        kurtosis.partial_cmp(&T::zero()) != Some(Ordering::Less),
        "kurtosis must be non-negative"
    );
    Moments {
        total,
        mean,
        variance,
        stddev,
        skewness,
        kurtosis,
    }
}

fn print_moments<T: IntervalNumber>(moments: &Moments<Interval<T>>, print_intervals: bool) {
    let Moments {
        total,
        mean,
        variance,
        stddev,
        skewness,
        kurtosis,
    } = moments;
    let pi = print_intervals;
    println!("Total measure:             Z {}", in_interval(total, pi));
    println!("Expected value:            E {}", in_interval(mean, pi));
    println!("Standard deviation:        σ {}", in_interval(stddev, pi));
    println!("Variance:                  V {}", in_interval(variance, pi));
    println!("Skewness (3rd std moment): S {}", in_interval(skewness, pi));
    println!("Kurtosis (4th std moment): K {}", in_interval(kurtosis, pi));
}

fn print_elapsed_message(start: Instant, text: &str, args: &CliArgs) {
    if !args.no_timing {
        let elapsed = start.elapsed().as_secs_f64();
        print!("{text}");
        if elapsed < 0.001 {
            println!("{elapsed:.6}s")
        } else if elapsed < 0.01 {
            println!("{elapsed:.5}s")
        } else if elapsed < 0.1 {
            println!("{elapsed:.4}s")
        } else {
            println!("{elapsed:.3}s")
        }
    }
}

fn print_json<T: Number>(
    moments_data: (&Moments<T>, Duration),
    probs_data: &Option<(Vec<T>, Duration)>,
    gf_translation_time: Duration,
    inference_time: Duration,
    args: &CliArgs,
    json_path: &PathBuf,
) -> std::io::Result<()> {
    let model_name = args.file_name.file_stem().unwrap().to_str().unwrap();
    let (moments_data, time_for_moments) = moments_data;
    let (probs_data, time_for_probs) = probs_data.clone().unwrap_or((vec![], Duration::default()));
    let Moments {
        total,
        mean,
        variance,
        stddev,
        skewness,
        kurtosis,
    } = moments_data;
    std::fs::write(
        json_path,
        format!(
            r#"
{{
    "model": "{model_name}",
    "system": "genfer",
    "time_gf_translation": {gf_translation_time},
    "total": {total},
    "mean": {mean},
    "variance": {variance},
    "stddev": {stddev},
    "skewness": {skewness},
    "kurtosis": {kurtosis},
    "time_moments": {time_for_moments},
    "masses": [{masses}],
    "time_probs": {time_for_probs},
    "time_infer": {total_time},
}}
"#,
            gf_translation_time = gf_translation_time.as_secs_f64(),
            time_for_moments = time_for_moments.as_secs_f64(),
            masses = probs_data
                .into_iter()
                .map(|x| format!("{x}, "))
                .collect::<String>(),
            time_for_probs = time_for_probs.as_secs_f64(),
            total_time = inference_time.as_secs_f64(),
        ),
    )
}

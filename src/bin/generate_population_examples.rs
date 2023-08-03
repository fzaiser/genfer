use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Binomial, Distribution, Poisson};

pub fn main() -> std::io::Result<()> {
    // modified means that there is an extra condition on the arrival rate
    for modified in [false, true] {
        for size in [50, 100, 200, 500, 1000, 2000] {
            generate_population_model(size, modified)?;
        }
    }
    Ok(())
}

fn generate_population_model(size: u32, modified: bool) -> std::io::Result<()> {
    use std::io::Write;
    let mut rng = StdRng::seed_from_u64(0);
    // Data taken from Winner et al. NeurIPS 2016
    let arrival_rate_fractions = [0.0257, 0.1163, 0.2104, 0.1504, 0.0428];
    let arrival_rates = arrival_rate_fractions
        .iter()
        .map(|&x| x * f64::from(size))
        .collect::<Vec<_>>();
    let survival_rates = [0.2636, 0.2636, 0.2636, 0.2636];
    let detection_prob = 0.2;
    let mut populations = [0; 5];
    let mut observations = [0; 5];
    populations[0] = Poisson::new(arrival_rates[0]).unwrap().sample(&mut rng) as u32;
    for i in 1..5 {
        let new_arrivals = Poisson::new(arrival_rates[i]).unwrap().sample(&mut rng) as u32;
        let survivors = Binomial::new(populations[i - 1].into(), survival_rates[i - 1])
            .unwrap()
            .sample(&mut rng) as u32;
        populations[i] = new_arrivals + survivors;
        observations[i] = Binomial::new(populations[i].into(), detection_prob)
            .unwrap()
            .sample(&mut rng) as u32;
    }
    for num_vars in 1..=4 {
        let program = population(
            &arrival_rates,
            &survival_rates,
            detection_prob,
            &observations,
            modified,
            num_vars,
        )
        .unwrap();
        let suffix = if modified { "_modified" } else { "" };
        let mut file = std::fs::File::create(format!(
            "examples/population/{size}_{num_vars}vars{suffix}.sgcl"
        ))?;
        file.write_all(program.as_bytes())?;
    }
    Ok(())
}

pub fn population(
    arrival_rates: &[f64],
    survival_rates: &[f64],
    detection_prob: f64,
    observations: &[u32],
    modified: bool,
    num_vars: usize,
) -> Result<String, std::fmt::Error> {
    use std::fmt::Write;
    let mut code = String::new();
    writeln!(&mut code, "population ~ Poisson({});", arrival_rates[0])?;
    for i in 0..survival_rates.len() {
        writeln!(&mut code)?;
        if num_vars >= 2 {
            if modified {
                writeln!(&mut code, "if 1 ~ Bernoulli(0.1) {{ arrivals ~ Poisson({}); }} else {{ arrivals ~ Poisson({}); }}", arrival_rates[i + 1] / 10.0, arrival_rates[i + 1])?;
            } else {
                writeln!(&mut code, "arrivals ~ Poisson({});", arrival_rates[i + 1])?;
            }
            if num_vars >= 4 {
                writeln!(
                    &mut code,
                    "survivors ~ Binomial(population, {});\npopulation := survivors;\npopulation += arrivals;",
                    survival_rates[i]
                )?;
            } else {
                writeln!(
                    &mut code,
                    "population ~ Binomial(population, {});\npopulation += arrivals;",
                    survival_rates[i]
                )?;
            }
        } else {
            writeln!(
                &mut code,
                "population ~ Binomial(population, {});",
                survival_rates[i],
            )?;
            if modified {
                writeln!(
                    &mut code,
                    "if 1 ~ Bernoulli(0.1) {{ population +~ Poisson({}); }} else {{ population +~ Poisson({}); }}",
                    arrival_rates[i + 1] / 10.0,
                    arrival_rates[i + 1],
                )?;
            } else {
                writeln!(
                    &mut code,
                    "population +~ Poisson({});",
                    arrival_rates[i + 1],
                )?;
            }
        }
        if num_vars >= 3 {
            writeln!(
                &mut code,
                "observed ~ Binomial(population, {});\nobserve observed = {};",
                detection_prob,
                observations[i + 1]
            )?;
        } else {
            writeln!(
                &mut code,
                "observe {} ~ Binomial(population, {});",
                observations[i + 1],
                detection_prob,
            )?;
        }
    }
    writeln!(&mut code)?;
    writeln!(&mut code, "return population")?;
    Ok(code)
}

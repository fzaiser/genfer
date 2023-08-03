use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Binomial, Distribution, Poisson};

pub fn main() -> std::io::Result<()> {
    for size in [50, 100, 200, 500, 1000, 2000] {
        generate_population_model(size)?;
    }
    Ok(())
}

fn sample_poisson(rate: f64, rng: &mut StdRng) -> u32 {
    Poisson::new(rate).unwrap().sample(rng) as u32
}

fn sample_binomial(n: u32, p: f64, rng: &mut StdRng) -> u32 {
    Binomial::new(n.into(), p).unwrap().sample(rng) as u32
}

fn generate_population_model(size: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut rng = StdRng::seed_from_u64(0);
    // Data adapted from Winner et al. NeurIPS 2016
    let arrival_rate_fractions = [
        (0.0257 * 0.9, 0.0257 * 0.1),
        (0.1163 * 0.9, 0.1163 * 0.1),
        (0.2104 * 0.9, 0.2104 * 0.1),
        (0.1504 * 0.9, 0.1504 * 0.1),
        (0.0428 * 0.9, 0.0428 * 0.1),
    ];
    let arrival_rates = arrival_rate_fractions
        .iter()
        .map(|&x| (x.0 * f64::from(size), x.1 * f64::from(size)))
        .collect::<Vec<_>>();
    let survival_rates = [
        (0.2636, 0.2636),
        (0.2636, 0.2636),
        (0.2636, 0.2636),
        (0.2636, 0.2636),
    ];
    let prob1to2 = 0.1;
    let detection_prob = (0.2, 0.2);
    let mut populations = [(0, 0); 5];
    let mut observations = [(0, 0); 5];
    populations[0] = (
        sample_poisson(arrival_rates[0].0, &mut rng),
        sample_poisson(arrival_rates[0].1, &mut rng),
    );
    for i in 1..5 {
        let new_arrivals = (
            sample_poisson(arrival_rates[i].0, &mut rng),
            sample_poisson(arrival_rates[i].1, &mut rng),
        );
        populations[i - 1].1 += sample_binomial(populations[i - 1].0, prob1to2, &mut rng);
        let survivors = (
            sample_binomial(
                populations[i - 1].0,
                survival_rates[i - 1].0 * (1.0 - prob1to2),
                &mut rng,
            ),
            sample_binomial(populations[i - 1].1, survival_rates[i - 1].1, &mut rng),
        );
        populations[i].0 = new_arrivals.0 + survivors.0;
        populations[i].1 = new_arrivals.1 + survivors.1;
        observations[i].0 = sample_binomial(populations[i].0, detection_prob.0, &mut rng);
        observations[i].1 = sample_binomial(populations[i].1, detection_prob.1, &mut rng);
    }
    let program = population(
        &arrival_rates,
        prob1to2,
        &survival_rates,
        detection_prob,
        &observations,
    )
    .unwrap();
    let mut file =
        std::fs::File::create(format!("examples/population/two_populations{size}.sgcl"))?;
    file.write_all(program.as_bytes())?;
    Ok(())
}

pub fn population(
    arrival_rates: &[(f64, f64)],
    prob1to2: f64, // Probability of mutating from population 1 to population 2
    survival_rates: &[(f64, f64)],
    detection_prob: (f64, f64),
    observations: &[(u32, u32)],
) -> Result<String, std::fmt::Error> {
    use std::fmt::Write;
    let mut code = String::new();
    writeln!(&mut code, "population1 ~ Poisson({});", arrival_rates[0].0)?;
    writeln!(&mut code, "population2 ~ Poisson({});", arrival_rates[0].1)?;
    for i in 0..survival_rates.len() {
        writeln!(&mut code)?;
        writeln!(
            &mut code,
            "population2 +~ Binomial(population1, {});\npopulation1 ~ Binomial(population1, {});\npopulation2 ~ Binomial(population2, {});",
            prob1to2,
            survival_rates[i].0 * (1.0 - prob1to2),
            survival_rates[i].1,
        )?;
        writeln!(
            &mut code,
            "population1 +~ Poisson({});\npopulation2 +~ Poisson({});",
            arrival_rates[i + 1].0,
            arrival_rates[i + 1].1,
        )?;
        writeln!(
            &mut code,
            "observe {} ~ Binomial(population1, {});\nobserve {} ~ Binomial(population2, {});",
            observations[i + 1].0,
            detection_prob.0,
            observations[i + 1].1,
            detection_prob.1,
        )?;
    }
    writeln!(&mut code)?;
    writeln!(&mut code, "return population2")?;
    Ok(code)
}

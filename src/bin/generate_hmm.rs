use std::fs::File;

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Bernoulli, Distribution, Poisson};

pub fn main() {
    let mut file = File::create("examples/hmm.sgcl").unwrap();
    // Simulate data:
    let mut rng = StdRng::seed_from_u64(0);
    let mut data = [-1; 30]; // should be 50, but Anglican can't seem to handle more than 30 (?)
    let mut state = 1;
    let rate1 = 0.5;
    let rate2 = 2.5;
    for data_point in &mut data {
        if state == 0 {
            *data_point = sample_poisson(rate1, &mut rng) as i32;
            state = sample_bernoulli(0.2, &mut rng);
        } else {
            *data_point = sample_poisson(rate2, &mut rng) as i32;
            state = sample_bernoulli(0.8, &mut rng);
        }
    }
    let data = &data[..];
    generate(&mut file, data).unwrap();
}

fn sample_poisson(rate: f64, rng: &mut StdRng) -> u32 {
    Poisson::new(rate).unwrap().sample(rng) as u32
}

fn sample_bernoulli(p: f64, rng: &mut StdRng) -> u32 {
    Bernoulli::new(p).unwrap().sample(rng) as u32
}

fn generate(file: &mut dyn std::io::Write, data: &[i32]) -> std::io::Result<()> {
    writeln!(file, "# data: {data:?}")?;
    writeln!(file)?;
    writeln!(file, "State := 1;")?;
    writeln!(file, "Rate1 ~ Geometric(0.1);")?;
    writeln!(file, "Rate2 ~ Geometric(0.1);")?;
    for &d in data.iter() {
        if d != -1 {
            writeln!(
                file,
                "if State = 0 {{
    observe {d} ~ Poisson(0.1 * Rate1);
    State ~ Bernoulli(0.2);
}} else {{
    observe {d} ~ Poisson(0.1 * Rate2);
    State ~ Bernoulli(0.8);
}}"
            )?;
        }
    }
    writeln!(file)?;
    writeln!(file, "return Rate2;")?;
    Ok(())
}

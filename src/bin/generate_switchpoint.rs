use std::fs::File;

pub fn main() {
    // Source: https://docs.pymc.io/en/v3/pymc-examples/examples/getting_started.html#Case-study-2:-Coal-mining-disasters
    // Years: 1851-1961
    let data = [
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4,
        2, 5, 2, 2, 3, 4, 2, 1, 3, -1, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2,
        0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, -1, 2, 1, 1, 1, 1, 2,
        4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    ];
    let data = &data[..];

    let filename = "examples/switchpoint.sgcl";
    let mut file = File::create(filename).unwrap();
    generate(&mut file, false, data).unwrap();

    let filename = "examples/cont_switchpoint.sgcl";
    let mut file = File::create(filename).unwrap();
    generate(&mut file, true, data).unwrap();
}

fn generate(file: &mut dyn std::io::Write, cont: bool, data: &[i32]) -> std::io::Result<()> {
    if cont {
        writeln!(file, "rate ~ Exponential(1);")?;
    } else {
        writeln!(file, "rate ~ Geometric(0.1);")?;
    }
    for switchpoint in 0..data.len() {
        writeln!(
            file,
            "if 1 ~ Bernoulli(1 / {}) {{",
            data.len() - switchpoint
        )?;
        generate_with_fixed_switchpoint(file, switchpoint, cont, data)?;
        writeln!(file, "switchpoint := {switchpoint};")?;
        write!(file, "}} else ")?;
    }
    writeln!(file, "{{}}")?;
    writeln!(file)?;
    writeln!(file, "return switchpoint;")?;
    Ok(())
}

fn generate_with_fixed_switchpoint(
    file: &mut dyn std::io::Write,
    switchpoint: usize,
    cont: bool,
    data: &[i32],
) -> std::io::Result<()> {
    for (i, &d) in data.iter().enumerate() {
        if switchpoint == i {
            if cont {
                writeln!(file, "rate ~ Exponential(1);")?;
            } else {
                writeln!(file, "rate ~ Geometric(0.1);")?;
            }
        }
        if d >= 0 {
            if cont {
                writeln!(file, "observe {d} ~ Poisson(rate);")?;
            } else {
                writeln!(file, "observe {d} ~ Poisson(0.1 * rate);")?;
            }
        }
    }
    Ok(())
}

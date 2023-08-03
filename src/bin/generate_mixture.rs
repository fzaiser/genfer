use std::fs::File;

pub fn main() {
    let mut file = File::create("examples/mixture.sgcl").unwrap();
    // Data Source: https://docs.pymc.io/en/v3/pymc-examples/examples/getting_started.html#Case-study-2:-Coal-mining-disasters
    // Years: 1851-1961
    let data = [
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4,
        2, 5, 2, 2, 3, 4, 2, 1, 3, -1, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2,
        0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, -1, 2, 1, 1, 1, 1, 2,
        4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    ];
    let data = &data[..];
    generate(&mut file, data).unwrap();
}

fn generate(file: &mut dyn std::io::Write, data: &[i32]) -> std::io::Result<()> {
    writeln!(file, "Rate1 ~ Geometric(0.1);")?;
    writeln!(file, "Rate2 ~ Geometric(0.1);")?;
    for &d in data.iter() {
        if d != -1 {
            writeln!(
                file,
                "if 1 ~ Bernoulli(0.5) {{
    observe {d} ~ Poisson(0.1 * Rate1);
}} else {{
    observe {d} ~ Poisson(0.1 * Rate2);
}}"
            )?;
        }
    }
    writeln!(file)?;
    writeln!(file, "return Rate1;")?;
    Ok(())
}

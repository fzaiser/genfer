use std::fs::File;
use std::io::Write;

pub fn main() -> std::io::Result<()> {
    generate_digits()?;
    Ok(())
}

fn generate_digits() -> std::io::Result<()> {
    let mut sgcl = File::create("benchmarks/baselines/digitRecognition.sgcl")?;
    let mut psi = File::create("benchmarks/baselines/digitRecognition.psi")?;
    let mut dice = File::create("benchmarks/baselines/digitRecognition.dice")?;
    let mut prodigy = File::create("benchmarks/baselines/digitRecognition.pgcl")?;

    let priors = std::fs::read_to_string("benchmarks/baselines/data/digitPriors.csv")?
        .trim()
        .split(',')
        .map(|x| x.trim().to_owned())
        .collect::<Vec<String>>();

    let observations = std::fs::read_to_string("benchmarks/baselines/data/digitObservations.csv")?
        .trim()
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect::<Vec<i32>>();

    let params = std::fs::read_to_string("benchmarks/baselines/data/digitParams.csv")?
        .trim()
        .lines()
        .map(|line| {
            line.split(',')
                .map(|x| x.trim().to_owned())
                .collect::<Vec<String>>()
        })
        .collect::<Vec<Vec<String>>>();

    writeln!(psi, "// flags: --dp")?;
    writeln!(psi, "def main() {{")?;

    writeln!(sgcl, "y ~ Categorical(0.098717, 0.11237, 0.0993, 0.10218, 0.097367, 0.09035, 0.098633, 0.10442, 0.097517, 0.09915);")?;
    writeln!(psi, "    y := categorical([98717/1000000, 11237/100000, 993/10000, 10218/100000, 97367/1000000, 9035/100000, 98633/1000000, 10442/100000, 97517/1000000, 9915/100000]);")?;
    writeln!(dice, "let y = discrete(0.098717, 0.11237, 0.0993, 0.10218, 0.097367, 0.09035, 0.098633, 0.10442, 0.097517, 0.09915) in")?;
    writeln!(
        prodigy,
        r"nat y;

tmp := bernoulli(98717/1000004);
if(tmp = 1) {{
    y := 0;
}} else {{
    tmp := bernoulli(112370/901287);
    if(tmp = 1) {{
        y := 1;
    }} else {{
        tmp := bernoulli(99300/788917);
        if(tmp = 1) {{
            y := 2;
        }} else {{
            tmp := bernoulli(102180/689617);
            if(tmp = 1) {{
                y := 3;
            }} else {{
                tmp := bernoulli(97367/587437);
                if(tmp = 1) {{
                    y := 4;
                }} else {{
                    tmp := bernoulli(90350/490070);
                    if(tmp = 1) {{
                        y := 5;
                    }} else {{
                        tmp := bernoulli(98633/399720);
                        if(tmp = 1) {{
                            y := 6;
                        }} else {{
                            tmp := bernoulli(104420/301087);
                            if(tmp = 1) {{
                                y := 7;
                            }} else {{
                                tmp := bernoulli(97517/196667);
                                if(tmp = 1) {{
                                    y := 8;
                                }} else {{
                                    y := 9;
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
"
    )?;

    for i in 0..priors.len() {
        writeln!(sgcl, "if y = {i} {{")?;
        writeln!(prodigy, "if(y = {i}) {{")?;
        writeln!(psi, "    if(y == {i}) {{")?;
        if i < priors.len() - 1 {
            writeln!(dice, "if y == int(4, {i}) then")?;
        }

        for (idx, obs) in observations.iter().enumerate() {
            let decimals = params[i][idx].strip_prefix("0.").unwrap();
            let numer = decimals.parse::<u32>().unwrap();
            let denom = 10u32.pow(decimals.len() as u32);
            writeln!(sgcl, "    observe {obs} ~ Bernoulli({});", params[i][idx])?;
            writeln!(prodigy, "    tmp := bernoulli({numer}/{denom});")?;
            writeln!(prodigy, "    observe(tmp = {obs});")?;
            writeln!(psi, "        observe(flip({numer}/{denom}) == {obs});")?;
            let not = if obs == &0 { "!" } else { "" };
            writeln!(dice, "let _ = observe {not}(flip {}) in", params[i][idx])?;
        }

        writeln!(sgcl, "}}")?;
        writeln!(prodigy, "}} else {{skip}}")?;
        writeln!(psi, "}}")?;
        writeln!(dice, "y")?;
        if i < priors.len() - 1 {
            write!(dice, "else ")?;
        }
    }

    writeln!(sgcl, "return y;")?;
    writeln!(prodigy, "\ntmp := 0;\n\n?Pr[y];")?;
    writeln!(psi, "    return y;")?;

    writeln!(psi, "}}")?;

    Ok(())
}

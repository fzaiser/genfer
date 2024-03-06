#!/usr/bin/env python3

# Generates the PSI and SGCL code for this model with the parameters and observations in `data/`

import os
from pathlib import Path
import sys


def generate_digit_recognition():
    sgcl = open("digitRecognition.sgcl", "w")
    psi = open("digitRecognition.psi", "w")
    dice = open("digitRecognition.dice", "w")
    prodigy = open("digitRecognition.pgcl", "w")

    priors = open("data/digitPriors.csv").read().strip().split(",")
    priors = [x.strip() for x in priors]

    observations = open("data/digitObservations.csv").read().strip().split(",")
    observations = [int(x.strip()) for x in observations]

    params = open("data/digitParams.csv").read().strip().split("\n")
    params = [x.split(",") for x in params]
    params = [[y.strip() for y in x] for x in params]

    psi.write("// flags: --dp\n")
    psi.write("def main() {\n")

    sgcl.write(
        "# skip integration test\ny ~ Categorical(0.098717, 0.11237, 0.0993, 0.10218, 0.097367, 0.09035, 0.098633, 0.10442, 0.097517, 0.09915);\n"
    )
    psi.write(
        "    y := categorical([98717/1000000, 11237/100000, 993/10000, 10218/100000, 97367/1000000, 9035/100000, 98633/1000000, 10442/100000, 97517/1000000, 9915/100000]);\n"
    )
    dice.write(
        "let y = discrete(0.098717, 0.11237, 0.0993, 0.10218, 0.097367, 0.09035, 0.098633, 0.10442, 0.097517, 0.09915) in\n"
    )
    prodigy.write(
        r"""nat y;

tmp := bernoulli(98717/1000004);
if(tmp = 1) {
    y := 0;
} else {
    tmp := bernoulli(112370/901287);
    if(tmp = 1) {
        y := 1;
    } else {
        tmp := bernoulli(99300/788917);
        if(tmp = 1) {
            y := 2;
        } else {
            tmp := bernoulli(102180/689617);
            if(tmp = 1) {
                y := 3;
            } else {
                tmp := bernoulli(97367/587437);
                if(tmp = 1) {
                    y := 4;
                } else {
                    tmp := bernoulli(90350/490070);
                    if(tmp = 1) {
                        y := 5;
                    } else {
                        tmp := bernoulli(98633/399720);
                        if(tmp = 1) {
                            y := 6;
                        } else {
                            tmp := bernoulli(104420/301087);
                            if(tmp = 1) {
                                y := 7;
                            } else {
                                tmp := bernoulli(97517/196667);
                                if(tmp = 1) {
                                    y := 8;
                                } else {
                                    y := 9;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

"""
    )

    for i in range(len(priors)):
        sgcl.write(f"if y = {i} {{\n")
        prodigy.write(f"if(y = {i}) {{\n")
        psi.write(f"    if(y == {i}) {{\n")
        if i < len(priors) - 1:
            dice.write(f"if y == int(4, {i}) then\n")

        for idx, obs in enumerate(observations):
            decimals = params[i][idx].removeprefix("0.")
            numer = int(decimals)
            denom = 10 ** len(decimals)
            sgcl.write(f"    observe {obs} ~ Bernoulli({params[i][idx]});\n")
            prodigy.write(f"    tmp := bernoulli({numer}/{denom});\n")
            prodigy.write(f"    observe(tmp = {obs});\n")
            psi.write(f"        observe(flip({numer}/{denom}) == {obs});\n")
            not_operator = "!" if obs == 0 else ""
            dice.write(f"let _ = observe {not_operator}(flip {params[i][idx]}) in\n")

        sgcl.write("}\n")
        prodigy.write("} else {skip}\n")
        psi.write("}\n")
        dice.write("y\n")
        if i < len(priors) - 1:
            dice.write("else ")

    sgcl.write("return y;\n")
    prodigy.write("\ntmp := 0;\n\n?Pr[y];\n")
    psi.write("    return y;\n")

    psi.write("}\n")


if __name__ == "__main__":
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    generate_digit_recognition()

#!/usr/bin/env python3

# Generates the PSI and SGCL code for this model with the parameters and observations in `data/`

import os
from pathlib import Path
import sys

def generate_clinical_trial():
    count_data = 100

    with open("data/clinicalTrialControl.csv") as control_file:
        control_data = [int(x) for x in control_file.read().strip().split(", ")]
    
    with open("data/clinicalTrialTreated.csv") as treated_file:
        treated_data = [int(x) for x in treated_file.read().strip().split(", ")]

    assert len(control_data) == len(treated_data)
    total_count = len(control_data)

    control_data = control_data[:count_data]
    treated_data = treated_data[:count_data]

    with open("clinicalTrial.rational.sgcl", "w") as sgcl, open("clinicalTrial.psi", "w") as psi:
        psi.write(f"// Clinical trial data set, using {count_data} of {total_count} data points\n")
        sgcl.write("# skip integration test\n")
        sgcl.write(f"# Clinical trial data set, using {count_data} of {total_count} data points\n")

        psi.write("def main() {\n")
        psi.write("    isEffective := flip(1/2);\n")
        sgcl.write("isEffective ~ Bernoulli(1/2);\n")

        psi.write("    probControl := 0;\n")
        sgcl.write("probControl := 0;\n")

        psi.write("    probTreated := 0;\n")
        sgcl.write("probTreated := 0;\n")

        psi.write("    probAll := 0;\n")
        sgcl.write("probAll := 0;\n")

        psi.write("    if isEffective {\n")
        sgcl.write("if isEffective = 1 {\n")

        psi.write("        probControl = uniform(0, 1);\n")
        sgcl.write("    probControl ~ UniformCont(0, 1);\n")

        psi.write("        probTreated = uniform(0, 1);\n")
        sgcl.write("    probTreated ~ UniformCont(0, 1);\n")

        clinical_trial_loop(sgcl, psi, "probControl", control_data, "probTreated", treated_data)

        psi.write("    } else {\n")
        sgcl.write("} else {\n")

        psi.write("        probAll = uniform(0, 1);\n")
        sgcl.write("    probAll ~ UniformCont(0, 1);\n")

        clinical_trial_loop(sgcl, psi, "probAll", control_data, "probAll", treated_data)

        psi.write("    }\n")
        sgcl.write("}\n")

        psi.write("    return isEffective;\n")
        sgcl.write("return isEffective;\n")

        psi.write("}\n")

    sgcl_text = open("clinicalTrial.rational.sgcl").read()
    with open("clinicalTrial.sgcl", "w") as sgcl:
        sgcl.write("# skip integration test, flags: --precision 400\n")
        sgcl.write(sgcl_text)

def clinical_trial_loop(sgcl, psi, prob_control, control_data, prob_treated, treated_data):
    assert len(control_data) == len(treated_data)
    for data in control_data:
        sgcl.write(f"    observe {data} ~ Bernoulli({prob_control});\n")
        psi.write(f"        observe({data} = flip({prob_control}));\n")
    
    sgcl.write("\n")
    psi.write("\n")
    
    for data in treated_data:
        sgcl.write(f"    observe {data} ~ Bernoulli({prob_treated});\n")
        psi.write(f"        observe({data} = flip({prob_treated}));\n")

if __name__ == "__main__":
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    generate_clinical_trial()
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Fix matplotlib fonttype:
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def hitting_time_prob(time, prob_towards_goal, distance=1):
    """Probability of an asymmetric random walk hitting the point `dist` away from the origin in `time` steps.

    See here: https://math.stackexchange.com/questions/2794960/distribution-of-stopping-time-for-possibly-biased-random-walk
    """
    assert 0 < prob_towards_goal < 1
    assert distance > 0
    n = time
    p = prob_towards_goal
    a = distance
    if n >= a and (n + a) % 2 == 0:
        return (
            a
            / n
            * math.comb(n, (n - a) // 2)
            * (p) ** ((n + a) / 2)
            * (1 - p) ** ((n - a) / 2)
        )
    else:
        return 0


def parse_output(output_str):
    bounds = []
    for line in output_str.strip().split("\n"):
        bound = re.search(r"p\(\d+\) ∈ \[([\d.e-]+), ([\d.e-]+)\]", line)
        if bound:
            bounds.append((float(bound.group(1)), float(bound.group(2))))
        bound = re.search(r"p\(\d+\) = ([\d.e-]+)", line)
        if bound:
            bounds.append((float(bound.group(1)), float(bound.group(1))))
    time = re.search(r"Total time: ([\d.]+) ?s", output_str)
    time = float(time.group(1)) if time else None
    return bounds, time


def plot_benchmark(name, indices, exact, lin_thresh, tick_factor):
    tail_output = Path(f"outputs/{name}_bound_tail.txt").read_text()
    geom_bound_output = Path(f"outputs/{name}_bound_probs.txt").read_text()
    residual_output = Path(f"outputs/{name}_residual.txt").read_text()

    # Extracted bounds for residual_output
    residual_bounds, residual_time = parse_output(residual_output)
    count = len(residual_bounds)
    residual_bounds = [residual_bounds[i] for i in indices]
    residual_lowers = [lower for lower, _ in residual_bounds]
    residual_uppers = [upper for _, upper in residual_bounds]

    # Extracted bounds for geom_bound_output
    geom_bounds, geom_time = parse_output(geom_bound_output)
    assert len(geom_bounds) == count
    geom_bounds = [geom_bounds[i] for i in indices]
    geom_bound_lowers = [lower for lower, _ in geom_bounds]
    geom_bound_uppers = [upper for _, upper in geom_bounds]

    # Extracted bounds for geom_bound tail output
    tail_bounds, tail_time = parse_output(tail_output)
    assert len(tail_bounds) == count
    tail_bounds = [tail_bounds[i] for i in indices]
    tail_uppers = [upper for _, upper in tail_bounds]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot residual bounds:
    ax.plot(indices, residual_lowers, "r--", marker="|", alpha=0.5, linewidth=1)
    ax.plot(indices, residual_uppers, "r-", marker="|", alpha=0.5, linewidth=1)
    ax.fill_between(
        indices,
        residual_lowers,
        residual_uppers,
        color="red",
        alpha=0.2,
        label=f"Resid. mass bound ({residual_time:.2g} s)",
    )

    # Plot geom_bound bounds optimizing the bound on the total mass:
    ax.plot(indices, geom_bound_lowers, "b-", marker="|", alpha=0.5, linewidth=1)
    ax.plot(indices, geom_bound_uppers, "b--", marker="|", alpha=0.5, linewidth=1)
    ax.fill_between(
        indices,
        geom_bound_lowers,
        geom_bound_uppers,
        color="blue",
        alpha=0.2,
        label=f"Geom. bound, mass-opt. ({geom_time:.2g} s)",
    )

    # Plot tail bounds optimizing the bound on the tail decay rate:
    ax.plot(
        indices,
        tail_uppers,
        "k-",
        marker="|",
        alpha=0.5,
        linewidth=1,
        label=f"Geom. bound, tail-opt. ({tail_time:.2g} s)",
    )

    if exact is not None:
        # Plot exact masses:
        ax.scatter(
            indices,
            exact,
            marker="x",
            color="green",
            zorder=5,
            s=20,
            label="Exact probability",
        )

    # Setting symmetrical logarithmic scale
    ax.set_yscale("symlog", linthresh=lin_thresh)

    # Customizing the y-axis labels
    ax.set_ylim(bottom=0, top=1)
    yticks = [0, lin_thresh]
    while yticks[-1] < 0.99:
        yticks.append(yticks[-1] * tick_factor)
    yticks.append(1)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    plt.xlabel("Result value")
    plt.ylabel("Probability Mass")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # save as pdf
    plt.savefig(f"plot_{name}.pdf", bbox_inches="tight")


def plot_geo():
    indices = list(range(200))[1::2]
    plot_benchmark(
        "geo",
        indices=indices,
        exact=[0.5 ** (i + 1) for i in indices],
        lin_thresh=1e-40,
        tick_factor=1e5,
    )


def plot_asym_rw():
    indices = list(range(300))[1::2]
    plot_benchmark(
        "asym_rw",
        indices=indices,
        exact=[hitting_time_prob(i, 3 / 4) for i in indices],
        lin_thresh=1e-14,
        tick_factor=1e2,
    )


def plot_die_paradox():
    indices = list(range(100))[1:]
    plot_benchmark(
        "die_paradox",
        indices=indices,
        exact=[2 / 3**i if i > 0 else 0 for i in indices],
        lin_thresh=1e-48,
        tick_factor=1e8,
    )


def plot_coupon_collector():
    indices = list(range(300))[::3]
    plot_benchmark(
        "coupon_collector",
        indices=indices,
        exact=None,
        lin_thresh=1e-12,
        tick_factor=1e2,
    )


def plot_herman():
    indices = list(range(200))[1::2]
    plot_benchmark(
        "herman",
        indices=indices,
        exact=None,
        lin_thresh=1e-24,
        tick_factor=1e4,
    )


if __name__ == "__main__":
    plot_geo()
    plot_asym_rw()
    plot_die_paradox()
    plot_coupon_collector()
    plot_herman()

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from util import parse_output
from pathlib import Path

tail_output = Path("outputs/die_paradox_bound_tail.txt").read_text()

geom_bound_output = Path("outputs/die_paradox_bound_probs.txt").read_text()

residual_output = Path("outputs/die_paradox_residual.txt").read_text()                       

indices = list(range(200))[1:100]

# Extracted bounds for residual_output
residual_bounds = parse_output(residual_output)[1:100]
residual_lowers = [lower for lower, _ in residual_bounds]
residual_uppers = [upper for _, upper in residual_bounds]

# Extracted bounds for geom_bound_output
geom_bounds = parse_output(geom_bound_output)[1:100]
geom_bound_lowers = [lower for lower, _ in geom_bounds]
geom_bound_uppers = [upper for _, upper in geom_bounds]

# Extracted bounds for geom_bound tail output
tail_bounds = parse_output(tail_output)[1:100]
tail_uppers = [upper for _, upper in tail_bounds]

# Exact solution
exact = [2/3**i if i > 0 else 0 for i in indices]

fig, ax = plt.subplots(figsize=(6, 4))

# Plot residual bounds:
ax.plot(indices, residual_lowers, 'r--', marker='|', alpha=0.5, linewidth=1)
ax.plot(indices, residual_uppers, 'r-', marker='|', alpha=0.5, linewidth=1)
ax.fill_between(indices, residual_lowers, residual_uppers, color='red', alpha=0.2, label='Residual Mass Bounds')

# Plot geom_bound bounds:
ax.plot(indices, geom_bound_lowers, 'b-', marker='|', alpha=0.5, linewidth=1)
ax.plot(indices, geom_bound_uppers, 'b--', marker='|', alpha=0.5, linewidth=1)
ax.fill_between(indices, geom_bound_lowers, geom_bound_uppers, color='blue', alpha=0.2, label='Geometric Bound')

# Plot tail bounds:
ax.plot(indices, tail_uppers, 'k-', marker='|', alpha=0.5, linewidth=1, label='Geometric Bound (tail-optimized)')

# Plot exact masses:
ax.scatter(indices, exact, marker='x', color='green', zorder=5, s=20, label='Exact probability')

# Setting symmetrical logarithmic scale
linthresh = 1e-48
ax.set_yscale('symlog', linthresh=linthresh)

# Customizing the y-axis labels
ax.set_ylim(bottom=0, top=1)
ax.set_yticks([0, 1e-48, 1e-40, 1e-32, 1e-24, 1e-16, 1e-8, 1])
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

plt.xlabel('Result value')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# save as pdf
plt.savefig('plot_die_paradox.pdf', bbox_inches='tight')

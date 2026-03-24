"""
Basic Solow model: alternative steady-state diagram
Equation: \delta k_t = s A f(k_t), with f(k_t) = k_t^\alpha

Horizontal axis: k_t
Vertical axis: value

Output:
basic_solow_alternative.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# Output to figuretable/solow (sibling of code_for_figure)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), "figuretable", "solow")
os.makedirs(output_dir, exist_ok=True)

# Parameters
s = 0.3
A = 2
alpha = 0.2
delta = 0.8

# Steady state from delta k = s A k^alpha
k_star = (s * A / delta) ** (1 / (1 - alpha))

# Grid
k_max = 1.3 * k_star
k_grid = np.linspace(0, k_max, 500)

# Curves
investment = s * A * (k_grid ** alpha)
break_even = delta * k_grid

# Figure
fig, ax = plt.subplots(figsize=(6, 6))

# Plot curves
ax.plot(
    k_grid, investment,
    color="#1f77b4", lw=2.4,
    label=r"$sAf(k_t)$"
)
ax.plot(
    k_grid, break_even,
    color="black", lw=1.8,
    label=r"$\delta k_t$"
)

# Steady state guides
ax.axvline(k_star, color="0.65", linestyle=(0, (4, 3)), lw=1.0)
ax.axhline(delta * k_star, color="0.65", linestyle=(0, (4, 3)), lw=1.0)
ax.plot(k_star, delta * k_star, "o", color="#b22222", ms=6, zorder=5)

# Axis limits
ax.set_xlim(0, k_max * 1.02)
ax.set_ylim(0, max(investment.max(), break_even.max()) * 1.08)

# Axis labels at ends
ax.set_xlabel(r"$k_t$", fontsize=12)
ax.xaxis.set_label_coords(1.0, -0.04)


# Labels for steady state
ax.text(
    k_star, -0.08, r"$k^*$",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top", fontsize=12, clip_on=False
)

# Style
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

plt.tight_layout()

outpath = os.path.join(output_dir, "basic_solow_alternative.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {outpath} (k* = {k_star:.3f})")
"""
Basic Solow model: phase diagram
Equation: k_{t+1} = s A f(k_t) + (1 - delta) k_t, with f(k_t) = k_t^alpha
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Output to figuretable folder (sibling of code_for_figure)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), "figuretable")
os.makedirs(output_dir, exist_ok=True)

# Parameters
s = 0.3
A = 2
alpha = 0.2
delta = 0.8

# Steady state
k_star = (s * A / delta) ** (1 / (1 - alpha))

# Grid for k_t
k_max = 1.3 * k_star
k_grid = np.linspace(1e-6, k_max, 500)

# Law of motion
k_next = s * A * (k_grid ** alpha) + (1 - delta) * k_grid

# Figure
fig, ax = plt.subplots(figsize=(6, 6))

# 45-degree line
ax.plot(k_grid, k_grid, color="black", lw=1.8, label=r"$k_{t+1}=k_t$")

# LoM curve
ax.plot(
    k_grid, k_next,
    color="#1f77b4", lw=2.4,
    label=r"$k_{t+1}=sAf(k_t)+(1-\delta)k_t$"
)

# Steady state guides
ax.axhline(y=k_star, color="0.65", linestyle=(0, (4, 3)), lw=1)
ax.axvline(x=k_star, color="0.65", linestyle=(0, (4, 3)), lw=1)
ax.plot(k_star, k_star, "o", color="#b22222", ms=6, zorder=5)

# Limits
ax.set_xlim(0, k_max * 1.02)
ax.set_ylim(0, k_max * 1.02)

# Axis labels at ends
ax.set_xlabel(r"$k_t$", fontsize=12)
ax.xaxis.set_label_coords(1.02, -0.04)

ax.set_ylabel(fr"$k_{{t+1}}$", fontsize=12)
ax.yaxis.set_label_coords(-0.06, 0.96)

# k* labels outside axes
ax.text(
    k_star, -0.08, r"$k^*$",
    transform=ax.get_xaxis_transform(),
    ha="center", fontsize=12, clip_on=False
)
ax.text(
    -0.08, k_star, r"$k^*$",
    transform=ax.get_yaxis_transform(),
    ha="center", va="center", fontsize=12, clip_on=False
)

# Style
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

plt.tight_layout()
outpath = os.path.join(output_dir, "01_basic_solow.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to {outpath} (k* = {k_star:.1f})")
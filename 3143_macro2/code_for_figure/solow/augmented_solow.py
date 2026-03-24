"""
Labour-augmented Solow model: equilibrium diagram

Law of motion:
    \hat{k}_{t+1} = [ s A f(\hat{k}_t) + (1-\delta)\hat{k}_t ] / [(1+z)(1+n)]

with
    f(\hat{k}_t) = \hat{k}_t^\alpha

Output:
    3143_macro2/figuretable/solow/augmented_solow.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# Output: 3143_macro2/figuretable/solow/ (same folder as other lecture figures)
script_dir = os.path.dirname(os.path.abspath(__file__))
macro2_dir = os.path.dirname(os.path.dirname(script_dir))  # code_for_figure/solow -> 3143_macro2
output_dir = os.path.join(macro2_dir, "figuretable", "solow")
os.makedirs(output_dir, exist_ok=True)

# Parameters
s = 0.3
A = 2.0
alpha = 0.2
delta = 0.8
z = 0.05
n = 0.02

g = (1 + z) * (1 + n)

# Steady state:
# k = [s A k^alpha + (1-delta)k] / [(1+z)(1+n)]
# => [(1+z)(1+n) - (1-delta)] k = s A k^alpha
# => [delta + z + n + zn] k = s A k^alpha
# => k^(1-alpha) = s A / [delta + z + n + zn]
k_star = (s * A / (g - (1 - delta))) ** (1 / (1 - alpha))

def lom(k):
    return (s * A * (k ** alpha) + (1 - delta) * k) / g

# Grid
k_max = 1.3 * k_star
k_grid = np.linspace(1e-6, k_max, 600)
k_next = lom(k_grid)

# Figure
fig, ax = plt.subplots(figsize=(6, 6))

# 45-degree line
ax.plot(k_grid, k_grid, color="black", lw=1.8, label=r"$\hat{k}_{t+1}=\hat{k}_t$")

# LoM
ax.plot(
    k_grid, k_next,
    color="#1f77b4", lw=2.4,
    label=r"$\hat{k}_{t+1}=\dfrac{sAf(\hat{k})+(1-\delta)\hat{k}_t}{(1+z)(1+n)}$"
)

# Steady state guides
ax.axvline(k_star, color="0.65", linestyle=(0, (4, 3)), lw=1.0)
ax.axhline(k_star, color="0.65", linestyle=(0, (4, 3)), lw=1.0)
ax.plot(k_star, k_star, "o", color="#b22222", ms=6, zorder=5)

# Limits
ax.set_xlim(0, 1.02 * k_max)
ax.set_ylim(0, 1.02 * k_max)

# Axis labels at ends
ax.set_xlabel(r"$\hat{k}_t$", fontsize=12)
ax.xaxis.set_label_coords(1.0, -0.04)

ax.set_ylabel(r"$\hat{k}_{t+1}$", fontsize=12)
ax.yaxis.set_label_coords(-0.06, 0.98)

# Steady-state labels
ax.text(
    k_star, -0.08, r"$\hat{k}^*$",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top", fontsize=12, clip_on=False
)
ax.text(
    -0.08, k_star, r"$\hat{k}^*$",
    transform=ax.get_yaxis_transform(),
    ha="right", va="center", fontsize=12, clip_on=False
)

# Style
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="upper left", fontsize=9, frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

plt.tight_layout()

outpath = os.path.join(output_dir, "augmented_solow.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {outpath} (k* = {k_star:.4f})")

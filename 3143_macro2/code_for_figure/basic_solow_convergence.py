"""
Basic Solow model: phase diagram with convergence path
Two panels:
(a) k_0 < k* converging from below
(b) k_0 > k* converging from above
"""

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), "figuretable")
os.makedirs(output_dir, exist_ok=True)

# Parameters
s = 0.3
A = 2
alpha = 0.2
delta = 0.8
k_star = (s * A / delta) ** (1 / (1 - alpha))


def lom(k):
    return s * A * (k ** alpha) + (1 - delta) * k


def build_path(k_0, n_steps):
    k_vals = [k_0]
    for _ in range(n_steps):
        k_vals.append(lom(k_vals[-1]))
    return k_vals


def draw_cobweb(ax, k_vals, title=""):
    n_steps = len(k_vals) - 1
    k_max = 1.22 * max(k_star, max(k_vals))
    k_grid = np.linspace(1e-6, k_max, 600)
    k_next = lom(k_grid)

    # Main curves
    line_45, = ax.plot(
        k_grid, k_grid,
        color="black", lw=1.8, alpha=0.9,
        label=r"$k_{t+1}=k_t$"
    )
    line_lom, = ax.plot(
        k_grid, k_next,
        color="#1f77b4", lw=2.4,
        label=r"$k_{t+1}=sAf(k_t)+(1-\delta)k_t$"
    )

    # Steady state guides
    ax.axvline(k_star, color="0.65", ls=(0, (4, 3)), lw=1.0, zorder=0)
    ax.axhline(k_star, color="0.65", ls=(0, (4, 3)), lw=1.0, zorder=0)
    ax.scatter([k_star], [k_star], s=28, color="#b22222", zorder=5)

    # Cobweb path
    for t in range(n_steps):
        x0 = k_vals[t]
        y0 = k_vals[t]
        x1 = k_vals[t]
        y1 = k_vals[t + 1]
        x2 = k_vals[t + 1]
        y2 = k_vals[t + 1]

        # vertical move to LoM
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#b22222"),
            zorder=4
        )
        # horizontal move to 45-degree line
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#b22222"),
            zorder=4
        )

        # nodes
        ax.scatter([x0, x1, x2], [y0, y1, y2], s=14, color="black", zorder=5)

    # Light tick-style projections only to x-axis for iterates
    tick_len = 0.03 * k_max
    for t, k in enumerate(k_vals):
        ax.plot([k, k], [0, tick_len], color="0.55", lw=0.9)
        ax.text(k, -0.055 * k_max, rf"$k_{{{t}}}$", ha="center", va="top", fontsize=9)

    # k* labels
    ax.text(k_star, -0.055 * k_max, r"$k^*$", ha="center", va="top", fontsize=10, color="0.25")
    ax.text(-0.035 * k_max, k_star, r"$k^*$", ha="right", va="center", fontsize=10, color="0.25")

    # Cosmetics
    ax.set_xlim(0, k_max)
    ax.set_ylim(0, k_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$k_t$", fontsize=11)
    ax.xaxis.set_label_coords(1.02, -0.04)
    ax.set_ylabel(r"$k_{t+1}$", fontsize=11)
    ax.yaxis.set_label_coords(-0.06, 0.96)
    ax.set_title(title, fontsize=11, pad=8)

    # Minimal spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    return line_45, line_lom


# Paths
# Panel (a): k0 < k*, show up to k3
k_vals_below = build_path(k_star * 0.35, n_steps=3)

# Panel (b): k0 > k*, show up to k2
k_vals_above = build_path(k_star * 1.5, n_steps=2)

# Figure
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))

line_45, line_lom = draw_cobweb(axes[0], k_vals_below, title=r"(a) $k_0<k^*$")
draw_cobweb(axes[1], k_vals_above, title=r"(b) $k_0>k^*$")

# Shared legend
fig.legend(
    handles=[line_45, line_lom],
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.02)
)

plt.subplots_adjust(top=0.82, wspace=0.28, bottom=0.17)

outpath = os.path.join(output_dir, "01_basic_solow_convergence.png")
plt.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {outpath}")
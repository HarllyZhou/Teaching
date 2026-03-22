"""
Basic Solow model: dynamics of capital over time
Horizontal axis: time
Vertical axis: capital k_t

Two panels:
(a) k_0 < k^* converging from below
(b) k_0 > k^* converging from above

Output:
01_basic_solow_k_dynamics.png
"""

import os
import matplotlib
matplotlib.use("Agg")
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


def lom(k):
    return s * A * (k ** alpha) + (1 - delta) * k


def build_path(k0, n_steps):
    k_vals = [k0]
    for _ in range(n_steps):
        k_vals.append(lom(k_vals[-1]))
    return np.array(k_vals)


def draw_panel(ax, k_vals, title=""):
    t = np.arange(len(k_vals))

    # Smooth curve using polynomial interpolation (NumPy only)
    deg = min(2, len(t) - 1)
    coef = np.polyfit(t, k_vals, deg)
    poly = np.poly1d(coef)

    t_dense = np.linspace(t.min(), t.max(), 200)
    k_smooth = poly(t_dense)

    # Plot smooth path and actual points
    line_kt, = ax.plot(t_dense, k_smooth, lw=2.2, color="#1f77b4", label=r"$k_t$")
    ax.plot(t, k_vals, "o", color="#1f77b4", ms=5)

    # Steady state line
    line_kstar = ax.axhline(
        k_star, color="0.5", linestyle=(0, (4, 3)), lw=1.2, label=r"$k^*$"
    )

    # Labels for k_i
    offset = 0.035 * max(k_star, np.max(k_vals))
    for i, kv in enumerate(k_vals):
        ax.text(t[i], kv + offset, rf"$k_{{{i}}}$", ha="center", va="bottom", fontsize=9)

    # Label for k*
    ax.text(
        1.01, k_star, r"$k^*$",
        transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=10, clip_on=False
    )

    # Styling
    ax.set_title(title, fontsize=11)
    ax.set_xlim(t[0] - 0.15, t[-1] + 0.15)
    ax.set_xticks(t)
    ax.set_xticklabels([rf"$t={i}$" for i in t], fontsize=9)
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    return line_kt, line_kstar


# Paths
# Panel (a): up to k_3
k_vals_below = build_path(k_star * 0.35, n_steps=3)

# Panel (b): up to k_2
k_vals_above = build_path(k_star * 1.5, n_steps=2)

# Common y-limits
y_min = min(np.min(k_vals_below), np.min(k_vals_above), k_star) * 0.9
y_max = max(np.max(k_vals_below), np.max(k_vals_above), k_star) * 1.1

# Figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

line_kt, line_kstar = draw_panel(axes[0], k_vals_below, title=r"(a) $k_0 < k^*$")
draw_panel(axes[1], k_vals_above, title=r"(b) $k_0 > k^*$")

for ax in axes:
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("time", fontsize=11)
    ax.xaxis.set_label_coords(1.0, -0.08)

axes[0].set_ylabel(r"$k_t$", fontsize=11)
axes[0].yaxis.set_label_coords(-0.08, 0.98)

# Shared legend
fig.legend(
    [line_kt, line_kstar],
    [r"$k_t$", r"$k^*$"],
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.02)
)

plt.subplots_adjust(top=0.82, bottom=0.18, wspace=0.18)

outpath = os.path.join(output_dir, "01_basic_solow_k_dynamics.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {outpath}")
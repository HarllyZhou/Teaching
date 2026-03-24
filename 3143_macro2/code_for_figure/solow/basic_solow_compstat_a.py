"""
Basic Solow model:
1. Comparative statics of an increase in productivity A in the LoM diagram
2. IRFs to a positive permanent shock to A

Outputs:
    basic_solow_compstat_a.png
    basic_solow_a_shock_irf.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Paths
# =========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), "figuretable", "solow")
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# Parameters
# =========================================================
alpha = 0.2
delta = 0.8
s = 0.30

A0 = 2.0
A1 = 2.8

shock_time = 4
T = 24

# =========================================================
# Model objects
# =========================================================
def k_ss(A):
    return (s * A / delta) ** (1 / (1 - alpha))

def lom(k, A):
    return s * A * (k ** alpha) + (1 - delta) * k

def ss_vals(A, k):
    y = A * (k ** alpha)
    iota = s * y
    c = y - iota
    w = A * (1 - alpha) * (k ** alpha)
    R = A * alpha * (k ** (alpha - 1))
    return y, c, iota, w, R

k0_ss = k_ss(A0)
k1_ss = k_ss(A1)

# =========================================================
# Figure 1: Comparative statics in LoM diagram
# =========================================================
k_max = 1.3 * k1_ss
k_grid = np.linspace(1e-6, k_max, 600)

lom0 = lom(k_grid, A0)
lom1 = lom(k_grid, A1)

fig, ax = plt.subplots(figsize=(6.4, 6.0))

# 45-degree line
ax.plot(k_grid, k_grid, color="black", lw=1.8, label=r"$k_{t+1}=k_t$")

# LoM curves
ax.plot(
    k_grid, lom0,
    color="#1f77b4", lw=2.2,
    label=rf"$k_{{t+1}}=s A_0 k_t^\alpha + (1-\delta)k_t$"
)
ax.plot(
    k_grid, lom1,
    color="#d62728", lw=2.2,
    label=rf"$k_{{t+1}}=s A_1 k_t^\alpha + (1-\delta)k_t$"
)

# Steady states
ax.axvline(k0_ss, color="#1f77b4", linestyle=(0, (4, 3)), lw=1.0)
ax.axvline(k1_ss, color="#d62728", linestyle=(0, (4, 3)), lw=1.0)
ax.axhline(k0_ss, color="#1f77b4", linestyle=(0, (4, 3)), lw=1.0, alpha=0.7)
ax.axhline(k1_ss, color="#d62728", linestyle=(0, (4, 3)), lw=1.0, alpha=0.7)

ax.plot(k0_ss, k0_ss, "o", color="#1f77b4", ms=6, zorder=5)
ax.plot(k1_ss, k1_ss, "o", color="#d62728", ms=6, zorder=5)

# Arrow indicating upward shift
k_arrow = 0.55 * k1_ss
ax.annotate(
    "",
    xy=(k_arrow, lom(k_arrow, A1)),
    xytext=(k_arrow, lom(k_arrow, A0)),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="0.35")
)
ax.text(
    k_arrow + 0.04 * k1_ss,
    0.5 * (lom(k_arrow, A0) + lom(k_arrow, A1)),
    r"$A \uparrow$",
    fontsize=10, color="0.25", va="center"
)

# Axis limits
ax.set_xlim(0, 1.02 * k_max)
ax.set_ylim(0, 1.02 * k_max)

# Axis labels at ends
ax.set_xlabel(r"$k_t$", fontsize=12)
ax.xaxis.set_label_coords(1.0, -0.04)

ax.set_ylabel(r"$k_{{t+1}}$", fontsize=12)
ax.yaxis.set_label_coords(-0.06, 0.98)

# Labels for steady states
ax.text(
    k0_ss, -0.08, r"$k_0^*$",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top", fontsize=11, clip_on=False, color="#1f77b4"
)
ax.text(
    k1_ss, -0.08, r"$k_1^*$",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top", fontsize=11, clip_on=False, color="#d62728"
)
ax.text(
    -0.08, k0_ss, r"$k_0^*$",
    transform=ax.get_yaxis_transform(),
    ha="right", va="center", fontsize=11, clip_on=False, color="#1f77b4"
)
ax.text(
    -0.08, k1_ss, r"$k_1^*$",
    transform=ax.get_yaxis_transform(),
    ha="right", va="center", fontsize=11, clip_on=False, color="#d62728"
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
outpath1 = os.path.join(output_dir, "basic_solow_compstat_a.png")
plt.savefig(outpath1, dpi=150, bbox_inches="tight")
plt.close()

# =========================================================
# Figure 2: IRFs to a permanent increase in A
# =========================================================
k = np.zeros(T + 1)
k[0] = k0_ss

for t in range(T):
    A_t = A0 if t < shock_time else A1
    k[t + 1] = lom(k[t], A_t)

t_grid = np.arange(T + 1)
A_path = np.where(t_grid < shock_time, A0, A1)

y = A_path * (k ** alpha)
iota = s * y
c = y - iota
w = A_path * (1 - alpha) * (k ** alpha)
R = A_path * alpha * (k ** (alpha - 1))

y0_ss, c0_ss, i0_ss, w0_ss, R0_ss = ss_vals(A0, k0_ss)
y1_ss, c1_ss, i1_ss, w1_ss, R1_ss = ss_vals(A1, k1_ss)

def draw_irf(ax, series, old_ss, new_ss, ylab):
    color_main = "#1f77b4"

    # Pre-shock: horizontal line from 0 to exactly shock_time
    ax.plot([0, shock_time], [old_ss, old_ss], color=color_main, lw=2.2)

    # Post-shock: connect points at and after shock_time by line segments
    t_post = t_grid[t_grid >= shock_time]
    x_post = series[t_grid >= shock_time]
    ax.plot(t_post, x_post, color=color_main, lw=2.2)
    ax.scatter(t_post, x_post, color=color_main, s=16, zorder=3)

    # Shock line
    ax.axvline(shock_time, color="0.60", linestyle=(0, (4, 3)), lw=1.0)

    # Steady-state reference lines
    ax.axhline(old_ss, color="0.75", linestyle=(0, (2, 2)), lw=1.0)
    ax.axhline(new_ss, color="0.45", linestyle=(0, (4, 3)), lw=1.0)

    # Steady-state labels
    ax.text(
        -0.03, old_ss, rf"${ylab}_0^*$",
        transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=10, clip_on=False
    )
    ax.text(
        1.01, new_ss, rf"${ylab}_1^*$",
        transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=10, clip_on=False
    )

    ax.set_xlim(0, T)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

draw_irf(axes[0, 0], k,    k0_ss, k1_ss, "k")
draw_irf(axes[0, 1], y,    y0_ss, y1_ss, "y")
draw_irf(axes[1, 0], c,    c0_ss, c1_ss, "c")
draw_irf(axes[1, 1], iota, i0_ss, i1_ss, r"\iota")
draw_irf(axes[2, 0], w,    w0_ss, w1_ss, "w")
draw_irf(axes[2, 1], R,    R0_ss, R1_ss, "R")

axes[0, 0].set_title(r"$k_t$", fontsize=11)
axes[0, 1].set_title(r"$y_t$", fontsize=11)
axes[1, 0].set_title(r"$c_t$", fontsize=11)
axes[1, 1].set_title(r"$\iota_t$", fontsize=11)
axes[2, 0].set_title(r"$w_t$", fontsize=11)
axes[2, 1].set_title(r"$R_t$", fontsize=11)

for ax in axes[2, :]:
    ax.set_xlabel("time", fontsize=11)
    ax.xaxis.set_label_coords(1.0, -0.08)

plt.subplots_adjust(hspace=0.35, wspace=0.22, bottom=0.10)

outpath2 = os.path.join(output_dir, "basic_solow_a_shock_irf.png")
plt.savefig(outpath2, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {outpath1}")
print(f"Saved to {outpath2}")
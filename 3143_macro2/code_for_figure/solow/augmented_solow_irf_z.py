"""
Labour-augmented Solow model: IRFs to a positive permanent shock in z

Law of motion:
    \hat{k}_{t+1} = [ s A \hat{k}_t^\alpha + (1-\delta)\hat{k}_t ] / [(1+z_t)(1+n)]

with
    f(\hat{k}_t) = \hat{k}_t^\alpha

IRFs plotted for:
    k_t = K_t / N_t
    y_t = Y_t / N_t
    c_t = C_t / N_t
    i_t = I_t / N_t
    w_t
    R_t

Output:
    3143_macro2/figuretable/solow/augemented_solow_irf_z.png
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
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
output_dir = os.path.join(project_root, "figuretable", "solow")
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# Parameters
# =========================================================
A = 2.0
alpha = 0.2
delta = 0.8
s = 0.3
n = 0.02

z0 = 0.05
z1 = 0.08

shock_time = 4
T = 24

# =========================================================
# Steady states in effective labor units
# =========================================================
def khat_ss(z):
    g = (1 + z) * (1 + n)
    return (s * A / (g - (1 - delta))) ** (1 / (1 - alpha))

khat0_ss = khat_ss(z0)
khat1_ss = khat_ss(z1)

# =========================================================
# Simulate transition
# =========================================================
t_grid = np.arange(T + 1)

# z_t applies to the transition from t to t+1
z_transition = np.array([z0 if t < shock_time else z1 for t in range(T)])

# Z_t path
Z = np.zeros(T + 1)
Z[0] = 1.0
for t in range(T):
    Z[t + 1] = (1 + z_transition[t]) * Z[t]

# \hat{k}_t path
khat = np.zeros(T + 1)
khat[0] = khat0_ss

for t in range(T):
    g_t = (1 + n) * (1 + z_transition[t])
    khat[t + 1] = (s * A * (khat[t] ** alpha) + (1 - delta) * khat[t]) / g_t

# =========================================================
# Per-capita variables
# =========================================================
# k_t = K_t/N_t = Z_t * \hat{k}_t
# y_t = Y_t/N_t = A Z_t \hat{k}_t^\alpha
# c_t = (1-s) y_t
# i_t = s y_t
# w_t = A (1-alpha) Z_t \hat{k}_t^\alpha
# R_t = A alpha \hat{k}_t^(alpha-1)

k = Z * khat
y = A * Z * (khat ** alpha)
iota = s * y
c = y - iota
w = A * (1 - alpha) * Z * (khat ** alpha)
R = A * alpha * (khat ** (alpha - 1))

# =========================================================
# Reference paths
# =========================================================
# Old BGP path under z0
Z0 = (1 + z0) ** t_grid
k0_path = Z0 * khat0_ss
y0_path = A * Z0 * (khat0_ss ** alpha)
c0_path = (1 - s) * y0_path
i0_path = s * y0_path
w0_path = A * (1 - alpha) * Z0 * (khat0_ss ** alpha)
R0_path = np.full(T + 1, A * alpha * (khat0_ss ** (alpha - 1)))

# New BGP path under z1, anchored at the shock date for visual comparison
Z1_anchor = np.zeros(T + 1)
for t in range(T + 1):
    if t <= shock_time:
        Z1_anchor[t] = np.nan
    else:
        Z1_anchor[t] = Z[shock_time] * ((1 + z1) ** (t - shock_time))

# include the shock point itself
Z1_anchor[shock_time] = Z[shock_time]

k1_path = Z1_anchor * khat1_ss
y1_path = A * Z1_anchor * (khat1_ss ** alpha)
c1_path = (1 - s) * y1_path
i1_path = s * y1_path
w1_path = A * (1 - alpha) * Z1_anchor * (khat1_ss ** alpha)
R1_path = np.full(T + 1, np.nan)
R1_path[shock_time:] = A * alpha * (khat1_ss ** (alpha - 1))

# =========================================================
# Plot helper
# =========================================================
def draw_irf(ax, series, old_path, new_path, ylab):
    color_main = "#1f77b4"

    # old BGP before shock
    ax.plot(t_grid[:shock_time + 1], old_path[:shock_time + 1], color=color_main, lw=2.2)

    # actual path from shock onward
    t_post = t_grid[t_grid >= shock_time]
    x_post = series[t_grid >= shock_time]
    ax.plot(t_post, x_post, color=color_main, lw=2.2)
    ax.scatter(t_post, x_post, color=color_main, s=16, zorder=3)

    # shock line
    ax.axvline(shock_time, color="0.60", linestyle=(0, (4, 3)), lw=1.0)

    # reference paths
    ax.plot(t_grid, old_path, color="0.75", linestyle=(0, (2, 2)), lw=1.0)
    ax.plot(t_grid, new_path, color="0.45", linestyle=(0, (4, 3)), lw=1.0)

    # labels at right edge
    if np.isfinite(old_path[-1]):
        ax.text(
            1.01, old_path[-1], rf"${ylab}_0^*$",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=10, clip_on=False, color="0.45"
        )
    if np.isfinite(new_path[-1]):
        ax.text(
            1.01, new_path[-1], rf"${ylab}_1^*$",
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

# =========================================================
# Figure
# =========================================================
fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

draw_irf(axes[0, 0], k,    k0_path, k1_path, "k")
draw_irf(axes[0, 1], y,    y0_path, y1_path, "y")
draw_irf(axes[1, 0], c,    c0_path, c1_path, "c")
draw_irf(axes[1, 1], iota, i0_path, i1_path, r"\iota")
draw_irf(axes[2, 0], w,    w0_path, w1_path, "w")
draw_irf(axes[2, 1], R,    R0_path, R1_path, "R")

axes[0, 0].set_title(r"$k_t = K_t/N_t$", fontsize=11)
axes[0, 1].set_title(r"$y_t = Y_t/N_t$", fontsize=11)
axes[1, 0].set_title(r"$c_t = C_t/N_t$", fontsize=11)
axes[1, 1].set_title(r"$\iota_t = I_t/N_t$", fontsize=11)
axes[2, 0].set_title(r"$w_t$", fontsize=11)
axes[2, 1].set_title(r"$R_t$", fontsize=11)

for ax in axes[2, :]:
    ax.set_xlabel("time", fontsize=11)
    ax.xaxis.set_label_coords(1.0, -0.08)

plt.subplots_adjust(hspace=0.35, wspace=0.22, bottom=0.10)

outpath = os.path.join(output_dir, "augmented_solow_irf_z.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {outpath}")
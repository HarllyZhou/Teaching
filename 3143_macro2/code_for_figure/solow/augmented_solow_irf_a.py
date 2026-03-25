"""
Labour-augmented Solow model: IRFs to a positive permanent shock in A

Law of motion:
    \hat{k}_{t+1} = [ s A_t \hat{k}_t^\alpha + (1-\delta)\hat{k}_t ] / [(1+z)(1+n)]

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
    3143_macro2/figuretable/solow/augemented_solow_irf_a.png
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
alpha = 0.2
delta = 0.8
s = 0.3
z = 0.05
n = 0.02

A0 = 2.0
A1 = 2.8

shock_time = 4
T = 24

g = (1 + z) * (1 + n)

# =========================================================
# Steady states in effective labor units
# =========================================================
def khat_ss(A):
    return (s * A / (g - (1 - delta))) ** (1 / (1 - alpha))

khat0_ss = khat_ss(A0)
khat1_ss = khat_ss(A1)

# =========================================================
# Simulate transition in effective labor units
# =========================================================
def lom(khat, A):
    return (s * A * (khat ** alpha) + (1 - delta) * khat) / g

khat = np.zeros(T + 1)
khat[0] = khat0_ss

for t in range(T):
    A_t = A0 if t < shock_time else A1
    khat[t + 1] = lom(khat[t], A_t)

t_grid = np.arange(T + 1)
A_path = np.where(t_grid < shock_time, A0, A1)
Z = (1 + z) ** t_grid

# =========================================================
# Per-capita variables
# =========================================================
# k_t = K_t / N_t = Z_t * khat_t
# y_t = Y_t / N_t = A_t Z_t khat_t^alpha
# c_t = (1-s) y_t
# i_t = s y_t
# w_t = A_t (1-alpha) Z_t khat_t^alpha
# R_t = A_t alpha khat_t^(alpha-1)

k = Z * khat
y = A_path * Z * (khat ** alpha)
iota = s * y
c = y - iota
w = A_path * (1 - alpha) * Z * (khat ** alpha)
R = A_path * alpha * (khat ** (alpha - 1))

# =========================================================
# Reference balanced-growth paths
# =========================================================
k0_path = Z * khat0_ss
k1_path = Z * khat1_ss

y0_path = A0 * Z * (khat0_ss ** alpha)
y1_path = A1 * Z * (khat1_ss ** alpha)

c0_path = (1 - s) * y0_path
c1_path = (1 - s) * y1_path

i0_path = s * y0_path
i1_path = s * y1_path

w0_path = A0 * (1 - alpha) * Z * (khat0_ss ** alpha)
w1_path = A1 * (1 - alpha) * Z * (khat1_ss ** alpha)

R0_path = np.full(T + 1, A0 * alpha * (khat0_ss ** (alpha - 1)))
R1_path = np.full(T + 1, A1 * alpha * (khat1_ss ** (alpha - 1)))

# =========================================================
# Plot helper
# =========================================================
def draw_irf(ax, series, old_path, new_path, ylab):
    color_main = "#1f77b4"

    # before shock: old BGP
    ax.plot(t_grid[:shock_time + 1], old_path[:shock_time + 1], color=color_main, lw=2.2)

    # after shock: actual transition
    t_post = t_grid[t_grid >= shock_time]
    x_post = series[t_grid >= shock_time]
    ax.plot(t_post, x_post, color=color_main, lw=2.2)
    ax.scatter(t_post, x_post, color=color_main, s=16, zorder=3)

    # shock line
    ax.axvline(shock_time, color="0.60", linestyle=(0, (4, 3)), lw=1.0)

    # old and new BGP reference paths
    ax.plot(t_grid, old_path, color="0.75", linestyle=(0, (2, 2)), lw=1.0)
    ax.plot(t_grid, new_path, color="0.45", linestyle=(0, (4, 3)), lw=1.0)

    # labels at right edge
    ax.text(
        1.01, old_path[-1], rf"${ylab}_0^*$",
        transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=10, clip_on=False, color="0.45"
    )
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

outpath = os.path.join(output_dir, "augmented_solow_irf_a.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {outpath}")
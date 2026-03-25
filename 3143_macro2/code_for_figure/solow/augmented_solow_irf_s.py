"""
Solow model with labor-augmenting technology:
IRFs to a positive permanent shock to the saving rate s

Model:
    K_{t+1} = (1-delta) K_t + I_t
    I_t = s Y_t
    Y_t = A F(K_t, Z_t N_t)
    Y_t = C_t + I_t
    R_t = A F_K(K_t, Z_t N_t)
    w_t = A Z_t F_N(K_t, Z_t N_t)
    N_t = (1+n)^t
    Z_t = (1+z)^t

Take Cobb-Douglas:
    F(K, ZN) = K^alpha (ZN)^(1-alpha)

IRFs plotted for:
    k_t = K_t / N_t
    y_t = Y_t / N_t
    c_t = C_t / N_t
    i_t = I_t / N_t
    w_t
    R_t

Output:
    01_solow_labour_augmented_s_shock_irf.png
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
output_dir = os.path.join(os.path.dirname(script_dir), "figuretable")
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# Parameters
# =========================================================
A = 2.0
alpha = 0.2
delta = 0.08
n = 0.02
z = 0.05

s0 = 0.30
s1 = 0.42

shock_time = 4
T = 28

# =========================================================
# Steady-state objects in effective labor units
# =========================================================
g = (1 + n) * (1 + z)

def k_hat_ss(s):
    # \hat{k}^{*} = [ sA / (g - (1-delta)) ]^(1/(1-alpha))
    return (s * A / (g - (1 - delta))) ** (1 / (1 - alpha))

def y_hat_from_k_hat(k_hat):
    return A * (k_hat ** alpha)

# old and new steady states in effective labor units
khat0_ss = k_hat_ss(s0)
khat1_ss = k_hat_ss(s1)

yhat0_ss = y_hat_from_k_hat(khat0_ss)
yhat1_ss = y_hat_from_k_hat(khat1_ss)

# =========================================================
# Simulate transition in effective labor units
# =========================================================
# law of motion:
# \hat{k}_{t+1} = [ s A \hat{k}_t^alpha + (1-delta)\hat{k}_t ] / [(1+n)(1+z)]
khat = np.zeros(T + 1)
khat[0] = khat0_ss

for t in range(T):
    s_t = s0 if t < shock_time else s1
    khat[t + 1] = (s_t * A * (khat[t] ** alpha) + (1 - delta) * khat[t]) / g

t_grid = np.arange(T + 1)
s_path = np.where(t_grid < shock_time, s0, s1)

# =========================================================
# Build per capita variables
# =========================================================
# Since Z_t = (1+z)^t and N_t = (1+n)^t:
# K_t = \hat{k}_t Z_t N_t
# y_t = Y_t/N_t = A Z_t \hat{k}_t^alpha
# c_t = (1-s_t) y_t
# i_t = s_t y_t
# w_t = A (1-alpha) Z_t \hat{k}_t^alpha
# R_t = A alpha \hat{k}_t^(alpha-1)

Z = (1 + z) ** t_grid

k = Z * khat
y = A * Z * (khat ** alpha)
iota = s_path * y
c = y - iota
w = A * (1 - alpha) * Z * (khat ** alpha)
R = A * alpha * (khat ** (alpha - 1))

# =========================================================
# Reference balanced-growth paths before and after the shock
# =========================================================
# For k, y, c, iota, w: grow at rate (1+z) along a BGP
# For R: constant along a BGP

k0_path = k0_ss = Z * khat0_ss
k1_path = Z * khat1_ss

y0_path = A * Z * (khat0_ss ** alpha)
y1_path = A * Z * (khat1_ss ** alpha)

c0_path = (1 - s0) * y0_path
c1_path = (1 - s1) * y1_path

i0_path = s0 * y0_path
i1_path = s1 * y1_path

w0_path = A * (1 - alpha) * Z * (khat0_ss ** alpha)
w1_path = A * (1 - alpha) * Z * (khat1_ss ** alpha)

R0_path = np.full(T + 1, A * alpha * (khat0_ss ** (alpha - 1)))
R1_path = np.full(T + 1, A * alpha * (khat1_ss ** (alpha - 1)))

# =========================================================
# Plot helper
# =========================================================
def draw_irf(ax, series, old_path, new_path, ylab):
    color_main = "#1f77b4"

    # before shock: follow old BGP exactly up to shock time
    ax.plot(t_grid[:shock_time + 1], old_path[:shock_time + 1], color=color_main, lw=2.2)

    # after shock: actual transition path
    t_post = t_grid[t_grid >= shock_time]
    x_post = series[t_grid >= shock_time]
    ax.plot(t_post, x_post, color=color_main, lw=2.2)
    ax.scatter(t_post, x_post, color=color_main, s=16, zorder=3)

    # shock time
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
# project root: 3143_macro2
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# output directory: 3143_macro2/figuretable/solow
output_dir = os.path.join(project_root, "figuretable", "solow")
os.makedirs(output_dir, exist_ok=True)

outpath = os.path.join(output_dir, "augmented_solow_irf_s.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {outpath}")
"""compare_LB_energy.py — LB spectrum cumulative energy share comparison across three discretizations
==============================================================
Figure 1: Cumulative energy share of u_true_proj - u_base on LB eigenbasis under NURBS (single curve)
Figure 2: Cumulative energy share of u_true_proj - u_test on LB eigenbasis for P1 / P2 / NURBS (three curves)
Plot style matches the second subplot in the source file exactly (cumulative energy + 90% threshold + cutoff rank markers)
"""

import numpy as np
from scipy.sparse.linalg import eigsh, spsolve
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import warnings, sys
import matplotlib
matplotlib.use('TkAgg')          # switch to 'Agg' for headless mode
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


from surface_model import load_p1_from_npz, load_p2_from_npz, load_nurbs_from_npz
from surface_fem_solver import (
    assemble_p1_matrices, assemble_p2_matrices,
    assemble_nurbs_matrices, assemble_nurbs_load,
    l2_error, l2_error_nurbs,
    solve_poisson_zero_mean, solve_helmholtz,
)
from pullback_mms import compute_mms_from_npz, corrupt_source_term, simulate_sensors

# ======================== Unified Parameters ========================
NURBS_NPZ = "model/wavy_torus_nurbs.npz"
P1_NPZ    = "model/fem_p1_result.npz"
P2_NPZ    = "model/fem_p2_result.npz"

N_GRID    = 1200
K1, K2    = 2.0, 2.0

C_TRUTH   = 2.0       # True PDE reaction coefficient
C_MODEL   = 0.0       # Model PDE reaction coefficient (pure diffusion)

N_SENSORS = 9
NOISE_STD = 0.5

WHITE_STD = 0.5
LF_AMP    = 0.05
LF_MODES  = 2

SEED      = 42
NQ        = 7          # NURBS quadrature order

np.random.seed(SEED)

# ======================== Shared Data ========================
print("=" * 60)
print("Shared data preparation: MMS + noisy source term")
print("=" * 60)

x, y, z, u_true_grid, f_clean = compute_mms_from_npz(
    NURBS_NPZ, n_grid=N_GRID, k1=K1, k2=K2, c=C_TRUTH
)
f_noisy, _, _ = corrupt_source_term(
    f_clean, N_GRID,
    white_std=WHITE_STD, low_freq_amplitude=LF_AMP,
    low_freq_modes=LF_MODES, rng=SEED,
)

u_vals = np.linspace(0.0, 1.0, N_GRID)
v_vals = np.linspace(0.0, 1.0, N_GRID)

print(f"[INFO] u_true ∈ [{u_true_grid.min():.4f}, {u_true_grid.max():.4f}]")
print(f"[INFO] f_noisy ∈ [{f_noisy.min():.4f}, {f_noisy.max():.4f}]")


# ======================== Helper Functions ========================
def extract_lb_modes(K_stiff, M_mass, k_search=80):
    """Extract Laplace-Beltrami eigenmodes (excluding zero modes)."""
    vals_all, Phi_all = eigsh(
        K_stiff, k=k_search + 1, M=M_mass, sigma=0.0, which='LM'
    )
    order = np.argsort(vals_all)
    vals_all = vals_all[order]
    Phi_all  = Phi_all[:, order]

    tol_eig = 1e-8 * np.abs(vals_all).max()
    mask_nz = np.abs(vals_all) > tol_eig
    vals = vals_all[mask_nz]
    Phi  = Phi_all[:, mask_nz]
    return vals, Phi


def modal_energy_Q(b, Phi, M_mass, vals, kappa2):
    """
    Compute Q-weighted modal energy of b on the LB spectrum.
      b_modal_j = phi_j^T M b
      energy_j  = (kappa^2 + lambda_j) * b_modal_j^2
    """
    b_modal = Phi.T @ (M_mass @ b)
    w_j     = kappa2 + vals
    energy  = w_j * b_modal ** 2
    return energy

def effective_rank(energy, threshold=0.9):
    tot = float(energy.sum())
    if tot <= 0: return len(energy)
    cum = np.cumsum(energy) / tot
    idx = np.where(cum >= threshold)[0]
    return int(idx[0]) + 1 if len(idx) > 0 else len(energy)

def cumshare(energy):
    tot = float(energy.sum())
    return np.cumsum(energy) / tot if tot > 0 else np.zeros_like(energy)


# ======================== P1 ========================
print("\n" + "=" * 60)
print("Processing P1 ...")
print("=" * 60)

p1 = load_p1_from_npz(P1_NPZ)
K_p1, M_p1 = assemble_p1_matrices(p1)
nodes_2d_p1 = p1.nodes_2d
print(f"  P1: {p1.n_u}×{p1.n_v}, nodes={K_p1.shape[0]}")

interp_fn_p1 = RegularGridInterpolator((u_vals, v_vals), f_noisy)
F_noisy_p1 = M_p1 @ interp_fn_p1(nodes_2d_p1)
u_base_p1, _ = solve_poisson_zero_mean(K_p1, M_p1, F_noisy_p1)

interp_fc_p1 = RegularGridInterpolator((u_vals, v_vals), f_clean)
F_clean_p1 = M_p1 @ interp_fc_p1(nodes_2d_p1)
u_test_p1 = solve_helmholtz(K_p1, M_p1, C_TRUTH, F_clean_p1)

interp_ut_p1 = RegularGridInterpolator((u_vals, v_vals), u_true_grid)
u_true_proj_p1 = spsolve(M_p1.tocsc(), M_p1 @ interp_ut_p1(nodes_2d_p1))

vals_p1, Phi_p1 = extract_lb_modes(K_p1, M_p1)
kappa2_p1 = 0.5 * vals_p1[0]

eQ_test_p1 = modal_energy_Q(u_true_proj_p1 - u_test_p1, Phi_p1, M_p1, vals_p1, kappa2_p1)


# ======================== P2 ========================
print("\n" + "=" * 60)
print("Processing P2 ...")
print("=" * 60)

p2 = load_p2_from_npz(P2_NPZ)
K_p2, M_p2 = assemble_p2_matrices(p2)
nodes_2d_p2 = p2.nodes_2d
print(f"  P2: {p2.n_u}×{p2.n_v}, nodes={K_p2.shape[0]}")

interp_fn_p2 = RegularGridInterpolator((u_vals, v_vals), f_noisy)
F_noisy_p2 = M_p2 @ interp_fn_p2(nodes_2d_p2)
u_base_p2, _ = solve_poisson_zero_mean(K_p2, M_p2, F_noisy_p2)

interp_fc_p2 = RegularGridInterpolator((u_vals, v_vals), f_clean)
F_clean_p2 = M_p2 @ interp_fc_p2(nodes_2d_p2)
u_test_p2 = solve_helmholtz(K_p2, M_p2, C_TRUTH, F_clean_p2)

interp_ut_p2 = RegularGridInterpolator((u_vals, v_vals), u_true_grid)
u_true_proj_p2 = spsolve(M_p2.tocsc(), M_p2 @ interp_ut_p2(nodes_2d_p2))

vals_p2, Phi_p2 = extract_lb_modes(K_p2, M_p2)
kappa2_p2 = 0.5 * vals_p2[0]

eQ_test_p2 = modal_energy_Q(u_true_proj_p2 - u_test_p2, Phi_p2, M_p2, vals_p2, kappa2_p2)


# ======================== NURBS ========================
print("\n" + "=" * 60)
print("Processing NURBS ...")
print("=" * 60)

def load_nurbs_elems(npz_path):
    nurbs = load_nurbs_from_npz(str(npz_path))
    n_ctrl_u = nurbs.ctrl_u
    n_ctrl_v = nurbs.ctrl_v
    deg_u = nurbs.degree_u
    deg_v = nurbs.degree_v
    U = nurbs.knotvector_u
    V = nurbs.knotvector_v
    unique_u = np.unique(U)
    unique_v = np.unique(V)
    elem_spans = []
    for i in range(len(unique_u) - 1):
        ua, ub = unique_u[i], unique_u[i + 1]
        if ub - ua < 1e-14: continue
        span_u = np.searchsorted(U, (ua + ub) / 2.0, side='right') - 1
        span_u = np.clip(span_u, deg_u, n_ctrl_u - 1)
        for j in range(len(unique_v) - 1):
            va, vb = unique_v[j], unique_v[j + 1]
            if vb - va < 1e-14: continue
            span_v = np.searchsorted(V, (va + vb) / 2.0, side='right') - 1
            span_v = np.clip(span_v, deg_v, n_ctrl_v - 1)
            elem_spans.append((span_u, span_v, ua, ub, va, vb))
    return nurbs, elem_spans, n_ctrl_u, n_ctrl_v

nurbs, elem_spans, n_ctrl_u, n_ctrl_v = load_nurbs_elems(NURBS_NPZ)
K_n, M_n = assemble_nurbs_matrices(nurbs, elem_spans, nq=NQ)
print(f"  NURBS: {n_ctrl_u}×{n_ctrl_v} ctrl pts, DOFs={K_n.shape[0]}")

f_noise_interp_n = RegularGridInterpolator((u_vals, v_vals), f_noisy, method='cubic')
F_noisy_n = assemble_nurbs_load(nurbs, elem_spans, f_noise_interp_n, nq=NQ)
u_base_n, _ = solve_poisson_zero_mean(K_n, M_n, F_noisy_n)

f_clean_interp_n = RegularGridInterpolator((u_vals, v_vals), f_clean, method='cubic')
F_clean_n = assemble_nurbs_load(nurbs, elem_spans, f_clean_interp_n, nq=NQ)
u_test_n = solve_helmholtz(K_n, M_n, C_TRUTH, F_clean_n)

u_true_interp_n = RegularGridInterpolator((u_vals, v_vals), u_true_grid, method='cubic')
F_true_proj_n = assemble_nurbs_load(nurbs, elem_spans, u_true_interp_n, nq=NQ)
u_true_proj_n = spsolve(M_n.tocsc(), F_true_proj_n)

vals_n, Phi_n = extract_lb_modes(K_n, M_n)
kappa2_n = 0.5 * vals_n[0]

eQ_base_n = modal_energy_Q(u_true_proj_n - u_base_n, Phi_n, M_n, vals_n, kappa2_n)
eQ_test_n = modal_energy_Q(u_true_proj_n - u_test_n, Phi_n, M_n, vals_n, kappa2_n)


# ======================== Plotting ========================
K_SHOW = 80   # show first K modes

# ---- Figure 1: NURBS, cumulative energy share of u_true_proj - u_base ----
fig1, ax1 = plt.subplots(figsize=(10, 6))
k1_plot = min(K_SHOW, len(eQ_base_n))
modes1 = np.arange(1, k1_plot + 1)
cum_base_n = cumshare(eQ_base_n)[:k1_plot]
r_base_n = effective_rank(eQ_base_n)

ax1.plot(modes1, cum_base_n, 'b-', linewidth=2, label='NURBS b_base')
ax1.axhline(y=0.9, color='g', linestyle='--', linewidth=1.5, label='90% Threshold')
ax1.axvline(x=r_base_n, color='b', linestyle=':', alpha=0.6)
ax1.plot(r_base_n, cum_base_n[r_base_n - 1], 'bo', markersize=9, label=f'r_base(Q) = {r_base_n}')

ax1.set_xlabel('LB Mode Index (j)', fontsize=11)
ax1.set_ylabel('Cumulative Energy Share', fontsize=11)
ax1.set_ylim([0, 1.05])
ax1.set_title('Cumulative Energy & Cut-off Rank (NURBS: u_true_proj - u_base)', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('fig1_nurbs_base_cumshare.png', dpi=200)
plt.show()

# ---- Figure 2: P1 / P2 / NURBS, cumulative energy share of u_true_proj - u_test ----
fig2, ax2 = plt.subplots(figsize=(10, 6))
k2_plot = min(K_SHOW, len(eQ_test_p1), len(eQ_test_p2), len(eQ_test_n))
modes2 = np.arange(1, k2_plot + 1)

cum_test_p1 = cumshare(eQ_test_p1)[:k2_plot]
cum_test_p2 = cumshare(eQ_test_p2)[:k2_plot]
cum_test_n  = cumshare(eQ_test_n)[:k2_plot]

r_test_p1 = effective_rank(eQ_test_p1)
r_test_p2 = effective_rank(eQ_test_p2)
r_test_n  = effective_rank(eQ_test_n)

ax2.plot(modes2, cum_test_p1, 'b-', linewidth=2, label='P1 b_test')
ax2.plot(modes2, cum_test_p2, 'r-', linewidth=2, label='P2 b_test')
ax2.plot(modes2, cum_test_n, 'g-', linewidth=2, label='NURBS b_test')

ax2.axhline(y=0.9, color='g', linestyle='--', linewidth=1.5, label='90% Threshold')

ax2.axvline(x=r_test_p1, color='b', linestyle=':', alpha=0.6)
ax2.plot(r_test_p1, cum_test_p1[r_test_p1 - 1], 'bo', markersize=9, label=f'r_P1(Q) = {r_test_p1}')

ax2.axvline(x=r_test_p2, color='r', linestyle=':', alpha=0.6)
ax2.plot(r_test_p2, cum_test_p2[r_test_p2 - 1], 'ro', markersize=9, label=f'r_P2(Q) = {r_test_p2}')

ax2.axvline(x=r_test_n, color='g', linestyle=':', alpha=0.6)
ax2.plot(r_test_n, cum_test_n[r_test_n - 1], 'go', markersize=9, label=f'r_NURBS(Q) = {r_test_n}')

ax2.set_xlabel('LB Mode Index (j)', fontsize=11)
ax2.set_ylabel('Cumulative Energy Share', fontsize=11)
ax2.set_ylim([0, 1.05])
ax2.set_title('Cumulative Energy & Cut-off Rank (u_true_proj - u_test)', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('fig2_three_methods_test_cumshare.png', dpi=200)
plt.show()

print("\n[DONE] Two cumulative energy figures have been generated and saved.")

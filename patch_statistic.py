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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


from surface_model import load_p1_from_npz, load_p2_from_npz, load_nurbs_from_npz
from surface_fem_solver import (
    assemble_p1_matrices, assemble_p2_matrices,
    assemble_nurbs_matrices, assemble_nurbs_load,
    l2_error, l2_error_nurbs,
    solve_poisson_dirichlet, solve_helmholtz,
)
from pullback_mms import compute_mms_from_npz, corrupt_source_term, simulate_sensors

# ======================== Unified Parameters ========================
NURBS_NPZ = "model/nurbs_patch.npz"
P1_NPZ    = "model/fem_p1_patch.npz"
P2_NPZ    = "model/fem_p2_patch.npz"

N_GRID    = 800
K1, K2    = 2.0, 2.0

C_TRUTH   = 2.0       # True PDE reaction coefficient
C_MODEL   = 0.0       # Model PDE reaction coefficient (pure diffusion)

N_SENSORS = 9
NOISE_STD = 0.5

WHITE_STD = 0.5
LF_AMP    = 0.05
LF_MODES  = 2

SEED      = 42
NQ        = 5          # NURBS quadrature order

np.random.seed(SEED)

# ======================== Shared Data ========================
print("=" * 60)
print("Shared data preparation: MMS + noisy source term")
print("=" * 60)

# pullback_mms computes f = -Delta u + c*u, to match -Delta u - c*u = f we pass -C_TRUTH
x, y, z, u_true_grid, f_clean = compute_mms_from_npz(
    NURBS_NPZ, n_grid=N_GRID, k1=K1, k2=K2, c=-C_TRUTH
)
f_noisy, _, _ = corrupt_source_term(
    f_clean, N_GRID,
    white_std=WHITE_STD, low_freq_amplitude=LF_AMP,
    low_freq_modes=LF_MODES, rng=SEED,
)

# Consistent with original Galerkin: use pad to avoid interpolation boundary extrapolation
pad = 1e-6
u_vals = np.linspace(pad, 1.0 - pad, N_GRID)
v_vals = np.linspace(pad, 1.0 - pad, N_GRID)

print(f"[INFO] u_true ∈ [{u_true_grid.min():.4f}, {u_true_grid.max():.4f}]")
print(f"[INFO] f_noisy ∈ [{f_noisy.min():.4f}, {f_noisy.max():.4f}]")


# ======================== Helper Functions ========================
def extract_lb_modes_dirichlet(K_full, M_full, interior_dofs, k_search=80):
    """Extract Laplace-Beltrami eigenmodes (solved on interior DOFs after imposing Dirichlet BC).

    Parameters
    ----------
    K_full, M_full : sparse matrices
        Stiffness and mass matrices on the full space.
    interior_dofs : ndarray
        Interior degree-of-freedom indices.
    k_search : int
        Number of eigenvalues requested.

    Returns
    -------
    vals : (k_actual,) ndarray — nonzero eigenvalues
    Phi  : (n_dofs, k_actual) ndarray — eigenvectors expanded back to full nodal space (border entries zero)
    """
    n_dofs = K_full.shape[0]
    K_int = K_full[interior_dofs, :][:, interior_dofs]
    M_int = M_full[interior_dofs, :][:, interior_dofs]

    vals_all, Phi_all = eigsh(
        K_int, k=k_search + 1, M=M_int, sigma=0.0, which='LM'
    )
    order = np.argsort(vals_all)
    vals_all = vals_all[order]
    Phi_all  = Phi_all[:, order]

    # Filter zero modes (Dirichlet BC should eliminate zero eigenvalues, but keep safety logic)
    tol_eig = 1e-8 * np.abs(vals_all).max()
    mask_nz = np.abs(vals_all) > tol_eig
    vals = vals_all[mask_nz]
    Phi_int = Phi_all[:, mask_nz]

    # Expand back to full space (border entries zero)
    Phi = np.zeros((n_dofs, len(vals)))
    Phi[interior_dofs, :] = Phi_int

    return vals, Phi


def l2_proj_interior(u_func, M_full, interior_dofs, n_nodes, interp):
    """L2 projection on interior DOFs only, border entries zero.

    Solves M_int * u_proj_int = M_int_rhs, where rhs comes from u_func nodal sampling.
    """
    u_nodes = interp  # input values are already sampled at nodes
    M_int = M_full[interior_dofs, :][:, interior_dofs].tocsc()
    F_int = M_full[interior_dofs, :] @ u_nodes
    u_proj = np.zeros(n_nodes)
    u_proj[interior_dofs] = spsolve(M_int, F_int)
    return u_proj


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
    if tot <= 0:
        return len(energy)
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
n_nodes_p1 = nodes_2d_p1.shape[0]
print(f"  P1: {p1.n_u}×{p1.n_v}, nodes={n_nodes_p1}")

# Extract boundary DOFs
u_coor_p1 = nodes_2d_p1[:, 0]
v_coor_p1 = nodes_2d_p1[:, 1]
tol_bc = 1e-8
bc_mask_p1 = (u_coor_p1 < tol_bc) | (u_coor_p1 > 1.0 - tol_bc) | (v_coor_p1 < tol_bc) | (v_coor_p1 > 1.0 - tol_bc)
bc_dofs_p1 = np.where(bc_mask_p1)[0]
interior_dofs_p1 = np.setdiff1d(np.arange(n_nodes_p1), bc_dofs_p1)
print(f"  P1 boundary DOFs: {len(bc_dofs_p1)}, interior DOFs: {len(interior_dofs_p1)}")

# Source term interpolation -> load vector
interp_f_noise = RegularGridInterpolator((u_vals, v_vals), f_noisy, bounds_error=False, fill_value=None)
f_noise_nodes_p1 = interp_f_noise(nodes_2d_p1)
F_noisy_p1 = M_p1 @ f_noise_nodes_p1

# Solve baseline state (Poisson, -Delta u = f_noise, homogeneous Dirichlet)
u_base_p1 = solve_poisson_dirichlet(K_p1, F_noisy_p1, bc_dofs=bc_dofs_p1, bc_vals=None)

# Solve truth model (Helmholtz, -Delta u - C_TRUTH*u = f_clean)
interp_f_clean = RegularGridInterpolator((u_vals, v_vals), f_clean, bounds_error=False, fill_value=None)
f_clean_nodes_p1 = interp_f_clean(nodes_2d_p1)
F_clean_p1 = M_p1 @ f_clean_nodes_p1
u_test_p1 = solve_helmholtz(K_p1, M_p1, -C_TRUTH, F_clean_p1, bc_dofs=bc_dofs_p1, bc_vals=None)

# L2 projection of u_true (on interior DOFs)
interp_u_true = RegularGridInterpolator((u_vals, v_vals), u_true_grid, bounds_error=False, fill_value=None)
u_true_nodes_p1 = interp_u_true(nodes_2d_p1)
u_true_proj_p1 = l2_proj_interior(u_true_grid, M_p1, interior_dofs_p1, n_nodes_p1, u_true_nodes_p1)

# Extract Dirichlet LB modes
vals_p1, Phi_p1 = extract_lb_modes_dirichlet(K_p1, M_p1, interior_dofs_p1, k_search=80)
kappa2_p1 = 0.5 * vals_p1[0]
print(f"  P1 LB modes: {len(vals_p1)} found, lambda_1={vals_p1[0]:.4e}, kappa^2={kappa2_p1:.4e}")

# Error energy
eQ_test_p1 = modal_energy_Q(u_true_proj_p1 - u_test_p1, Phi_p1, M_p1, vals_p1, kappa2_p1)


# ======================== P2 ========================
print("\n" + "=" * 60)
print("Processing P2 ...")
print("=" * 60)

p2 = load_p2_from_npz(P2_NPZ)
K_p2, M_p2 = assemble_p2_matrices(p2)
nodes_2d_p2 = p2.nodes_2d
n_nodes_p2 = nodes_2d_p2.shape[0]
print(f"  P2: {p2.n_u}×{p2.n_v}, nodes={n_nodes_p2}")

# Extract boundary DOFs
u_coor_p2 = nodes_2d_p2[:, 0]
v_coor_p2 = nodes_2d_p2[:, 1]
bc_mask_p2 = (u_coor_p2 < tol_bc) | (u_coor_p2 > 1.0 - tol_bc) | (v_coor_p2 < tol_bc) | (v_coor_p2 > 1.0 - tol_bc)
bc_dofs_p2 = np.where(bc_mask_p2)[0]
interior_dofs_p2 = np.setdiff1d(np.arange(n_nodes_p2), bc_dofs_p2)
print(f"  P2 boundary DOFs: {len(bc_dofs_p2)}, interior DOFs: {len(interior_dofs_p2)}")

# Source term interpolation
f_noise_nodes_p2 = interp_f_noise(nodes_2d_p2)
F_noisy_p2 = M_p2 @ f_noise_nodes_p2

u_base_p2 = solve_poisson_dirichlet(K_p2, F_noisy_p2, bc_dofs=bc_dofs_p2, bc_vals=None)

f_clean_nodes_p2 = interp_f_clean(nodes_2d_p2)
F_clean_p2 = M_p2 @ f_clean_nodes_p2
u_test_p2 = solve_helmholtz(K_p2, M_p2, -C_TRUTH, F_clean_p2, bc_dofs=bc_dofs_p2, bc_vals=None)

# L2 projection
u_true_nodes_p2 = interp_u_true(nodes_2d_p2)
u_true_proj_p2 = l2_proj_interior(u_true_grid, M_p2, interior_dofs_p2, n_nodes_p2, u_true_nodes_p2)

# Dirichlet LB modes
vals_p2, Phi_p2 = extract_lb_modes_dirichlet(K_p2, M_p2, interior_dofs_p2, k_search=80)
kappa2_p2 = 0.5 * vals_p2[0]
print(f"  P2 LB modes: {len(vals_p2)} found, lambda_1={vals_p2[0]:.4e}, kappa^2={kappa2_p2:.4e}")

eQ_test_p2 = modal_energy_Q(u_true_proj_p2 - u_test_p2, Phi_p2, M_p2, vals_p2, kappa2_p2)


# ======================== NURBS ========================
print("\n" + "=" * 60)
print("Processing NURBS ...")
print("=" * 60)

def load_nurbs_elems(npz_path):
    """Load NURBS model and extract element spans."""
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
        if ub - ua < 1e-14:
            continue
        span_u = np.searchsorted(U, (ua + ub) / 2.0, side='right') - 1
        span_u = np.clip(span_u, deg_u, n_ctrl_u - 1)
        for j in range(len(unique_v) - 1):
            va, vb = unique_v[j], unique_v[j + 1]
            if vb - va < 1e-14:
                continue
            span_v = np.searchsorted(V, (va + vb) / 2.0, side='right') - 1
            span_v = np.clip(span_v, deg_v, n_ctrl_v - 1)
            elem_spans.append((span_u, span_v, ua, ub, va, vb))
    return nurbs, elem_spans, n_ctrl_u, n_ctrl_v

nurbs, elem_spans, n_ctrl_u, n_ctrl_v = load_nurbs_elems(NURBS_NPZ)
K_n, M_n = assemble_nurbs_matrices(nurbs, elem_spans, nq=NQ)
n_dofs_n = n_ctrl_u * n_ctrl_v
print(f"  NURBS: {n_ctrl_u}×{n_ctrl_v} ctrl pts, DOFs={n_dofs_n}")

# Extract boundary DOFs (clamped/open knot vector, first/last control points correspond to boundary)
bc_dofs_n = []
for i in range(n_ctrl_u):
    for j in range(n_ctrl_v):
        if i == 0 or i == n_ctrl_u - 1 or j == 0 or j == n_ctrl_v - 1:
            bc_dofs_n.append(i * n_ctrl_v + j)
bc_dofs_n = np.array(bc_dofs_n)
interior_dofs_n = np.setdiff1d(np.arange(n_dofs_n), bc_dofs_n)
print(f"  NURBS boundary DOFs: {len(bc_dofs_n)}, interior DOFs: {len(interior_dofs_n)}")

# Source term interpolation -> load vector
f_noise_interp_n = RegularGridInterpolator((u_vals, v_vals), f_noisy, method='cubic', bounds_error=False, fill_value=None)
F_noisy_n = assemble_nurbs_load(nurbs, elem_spans, f_noise_interp_n, nq=NQ)
u_base_n = solve_poisson_dirichlet(K_n, F_noisy_n, bc_dofs=bc_dofs_n, bc_vals=None)

# Truth model
f_clean_interp_n = RegularGridInterpolator((u_vals, v_vals), f_clean, method='cubic', bounds_error=False, fill_value=None)
F_clean_n = assemble_nurbs_load(nurbs, elem_spans, f_clean_interp_n, nq=NQ)
u_test_n = solve_helmholtz(K_n, M_n, -C_TRUTH, F_clean_n, bc_dofs=bc_dofs_n, bc_vals=None)

# L2 projection of u_true (on interior DOFs)
u_true_interp_n = RegularGridInterpolator((u_vals, v_vals), u_true_grid, method='cubic', bounds_error=False, fill_value=None)
F_true_proj_n = assemble_nurbs_load(nurbs, elem_spans, u_true_interp_n, nq=NQ)
M_int_n = M_n[interior_dofs_n, :][:, interior_dofs_n].tocsc()
F_true_int_n = F_true_proj_n[interior_dofs_n]  # boundary load is zero, extract directly
u_true_proj_n = np.zeros(n_dofs_n)
u_true_proj_n[interior_dofs_n] = spsolve(M_int_n, F_true_int_n)

# Dirichlet LB modes
vals_n, Phi_n = extract_lb_modes_dirichlet(K_n, M_n, interior_dofs_n, k_search=80)
kappa2_n = 0.5 * vals_n[0]
print(f"  NURBS LB modes: {len(vals_n)} found, lambda_1={vals_n[0]:.4e}, kappa^2={kappa2_n:.4e}")

# Modal energy for two error vectors
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
plt.savefig('fig1_nurbs_base_cumshare_T22.png', dpi=200)
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
plt.savefig('fig2_three_methods_test_cumshare_T22.png', dpi=200)
plt.show()

print("\n[DONE] Two cumulative energy figures have been generated and saved.")

"""
statFEM_P2.py — P2 data assimilation: analytic estimation vs Optuna Bayesian optimization
==============================================================
1. Analytic hyperparameter estimation (using u_true, F_clean, F_noisy)
2. Optuna Bayesian optimization for theoretical upper bound
3. Compare assimilation performance of both approaches
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import csc_matrix, lil_matrix, identity, bmat
from scipy.interpolate import RegularGridInterpolator
import warnings, sys, time
from pathlib import Path
import os

warnings.filterwarnings('ignore')

# -- Path setup --
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from surface_model import load_nurbs_from_npz, load_p2_from_npz
from surface_fem_solver import (
    assemble_p2_matrices, l2_error, solve_poisson_zero_mean, solve_helmholtz,
)
from pullback_mms import compute_mms_from_npz, corrupt_source_term, simulate_sensors

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# Parameter settings
# =====================================================================
P2_NPZ    = "fem_p2_result.npz"
NURBS_NPZ = "wavy_torus_nurbs.npz"
N_GRID    = 800
K1, K2    = 2.0, 2.0

C_TRUTH   = 1.0            # True PDE reaction coefficient
C_MODEL   = 0.0            # Model PDE reaction coefficient (pure diffusion)

N_SENSORS = 36              # Number of observations
NOISE_STD = 0.5           # Observation noise

WHITE_STD = 0.5           # Source term white noise strength
LF_AMP    = 0.05           # Source term bias amplitude
LF_MODES  = 2              # Source term bias frequency count

R_VALUES  = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] # List of reduced-rank dimensions to test
N_TRIALS  = 100             # Optuna optimization trials

SEED = 42
np.random.seed(SEED)

# =====================================================================
# Step 1-5: Geometry loading, MMS, noisy source, sensors, matrix assembly (one-time)
# =====================================================================
print("=" * 60)
print("Step 1-5: Geometry loading and matrix assembly (one-time preprocessing)")
print("=" * 60)
p2 = load_p2_from_npz(P2_NPZ)
n_nodes = p2.nodes_2d.shape[0]
n_dof = n_nodes
K, M = assemble_p2_matrices(p2)
nodes_2d = p2.nodes_2d

# Generate truth and noise data
x, y, z, u_true_grid, f_clean = compute_mms_from_npz(
    NURBS_NPZ, n_grid=N_GRID, k1=K1, k2=K2, c=C_TRUTH
)
f_noisy, _, _ = corrupt_source_term(
    f_clean, N_GRID, white_std=WHITE_STD, low_freq_amplitude=LF_AMP,
    low_freq_modes=LF_MODES, rng=SEED,
)

# Generate sensor data
nurbs = load_nurbs_from_npz(NURBS_NPZ)
phys_coords, u_obs, u_true_sens, uv_sensors = simulate_sensors(
    nurbs, n_sensors=N_SENSORS, noise_std=NOISE_STD, k1=K1, k2=K2, rng=SEED,
)

# Interpolate to nodes
u_vals = np.linspace(0.0, 1.0, N_GRID)
v_vals = np.linspace(0.0, 1.0, N_GRID)
interp_f_clean = RegularGridInterpolator((u_vals, v_vals), f_clean)
interp_f_noise = RegularGridInterpolator((u_vals, v_vals), f_noisy)
interp_u_true  = RegularGridInterpolator((u_vals, v_vals), u_true_grid)

f_clean_nodes = interp_f_clean(nodes_2d)
f_noise_nodes = interp_f_noise(nodes_2d)
u_true_nodes  = interp_u_true(nodes_2d)

F_clean_vec = M @ f_clean_nodes
F_noisy_vec = M @ f_noise_nodes

# --- Helper functions ---
def weighted_mean(vec, M_mat):
    return np.dot(M_mat @ vec, np.ones(len(vec))) / np.sum(M_mat)

def weighted_var(vec, M_mat):
    m = weighted_mean(vec, M_mat)
    return weighted_mean((vec - m)**2, M_mat)

# --- Precompute eigenvalues and eigenvectors at max scale (avoid recomputation in loop) ---
MAX_R = max(R_VALUES)
print(f"  Precomputing eigenvalues (Max R = {MAX_R})...")
N_EIG_REQUEST = MAX_R + 2 

try:
    eigvals_all, Vr_all = eigsh(
        K,
        k=N_EIG_REQUEST,
        M=M,
        sigma=0.0,
        which='LM'
    )
except Exception as e:
    print(f"  [WARNING] eigenvalue computation failed (requested modes {N_EIG_REQUEST} > DOFs {n_dof}), try reducing count...")
    N_EIG_REQUEST = min(n_dof - 2, MAX_R + 2)
    eigvals_all, Vr_all = eigsh(
        K,
        k=N_EIG_REQUEST,
        M=M,
        sigma=0.0,
        which='LM'
    )

# shift-invert does not guarantee ascending order; explicit sort required
order = np.argsort(eigvals_all)
eigvals_all = eigvals_all[order]
Vr_all      = Vr_all[:, order]

# Remove zero eigenvalues (constant mode on closed surface)
tol_eig = 1e-8 * np.abs(eigvals_all).max()
mask_nz = np.abs(eigvals_all) > tol_eig
eigvals_nz_all = eigvals_all[mask_nz]
Vr_nz_all      = Vr_all[:, mask_nz]

print(f"  Precomputation done. Found {len(eigvals_nz_all)}  nonzero eigenvalues.")

# --- Baseline solution (zero-mean Poisson) ---
u_base, lambda_lm = solve_poisson_zero_mean(K, M, F_noisy_vec)
L2_err_base, L2_true = l2_error(u_base, u_true_nodes, M)
print(f"[Baseline] u_base vs u_true: L2 = {L2_err_base:.6e}, |u_true| = {L2_true:.6e}")

# --- Build observation operator H ---
def build_observation_operator(p2, uv_sensors):
    n_obs = uv_sensors.shape[0]
    n_u, n_v = p2.n_u, p2.n_v
    du, dv = 1.0 / n_u, 1.0 / n_v
    H_lil = lil_matrix((n_obs, p2.nodes_2d.shape[0]))
    for i in range(n_obs):
        u, v = uv_sensors[i]
        ci = int(np.clip(u / du, 0, n_u - 1))
        cj = int(np.clip(v / dv, 0, n_v - 1))
        xi = (u - ci * du) / du
        eta = (v - cj * dv) / dv
        t1 = xi >= eta
        la = np.where(t1, 1.0 - xi, 1.0 - eta)
        lb = np.where(t1, xi - eta, xi)
        lc = np.where(t1, eta, eta - xi)
        
        nv = (n_u + 1) * (n_v + 1)
        nh = n_u * (n_v + 1)
        nve = (n_u + 1) * n_v
        i00 = cj + ci * (n_v + 1)
        i10 = cj + (ci + 1) * (n_v + 1)
        i11 = (cj + 1) + (ci + 1) * (n_v + 1)
        i01 = (cj + 1) + ci * (n_v + 1)
        ih00 = nv + cj + ci * (n_v + 1)
        ih01 = nv + (cj + 1) + ci * (n_v + 1)
        iv10 = nv + nh + cj + (ci + 1) * n_v
        iv00 = nv + nh + cj + ci * n_v
        id00 = nv + nh + nve + cj + ci * n_v
        
        d1 = i00
        d2 = np.where(t1, i10, i11)
        d3 = np.where(t1, i11, i01)
        d4 = np.where(t1, ih00, id00)
        d5 = np.where(t1, iv10, ih01)
        d6 = np.where(t1, id00, iv00)
        
        N1 = la * (2*la - 1); N2 = lb * (2*lb - 1); N3 = lc * (2*lc - 1)
        N4 = 4*la*lb; N5 = 4*lb*lc; N6 = 4*la*lc
        
        H_lil[i, d1] = N1; H_lil[i, d2] = N2; H_lil[i, d3] = N3
        H_lil[i, d4] = N4; H_lil[i, d5] = N5; H_lil[i, d6] = N6
    return H_lil.tocsr()

H = build_observation_operator(p2, uv_sensors)
sigma2_y_fixed = NOISE_STD ** 2

# =====================================================================
# Step 6: Analytic hyperparameter estimation
# =====================================================================
def estimate_hyperparams(K, M, u_true, F_noisy, c_truth, Vr, eigvals_r):
    sigma2_y = sigma2_y_fixed
    res = F_noisy - K @ u_true
    sigma2_me = weighted_mean(res**2, M)
    var_u = weighted_var(u_true, M)
    lam1 = eigvals_r[0] # Smallest nonzero eigenvalue under current R
    kappa2 = max(0.5 * lam1, 1e-4)
    kappa = np.sqrt(kappa2)
    tau = var_u * (kappa2 + lam1)

    u_true_anom = u_true - weighted_mean(u_true, M)
    alpha_true = Vr.T @ (M @ u_true_anom)
    var_alpha = np.abs(alpha_true)**2 + 1e-8
    Sigma_alpha_inv = np.diag(1.0 / var_alpha)

    return {
        'sigma2_y': sigma2_y, 'sigma2_me': sigma2_me,
        'kappa': kappa, 'tau': tau,
        'Sigma_alpha_inv': Sigma_alpha_inv
    }

# =====================================================================
# Step 7: Data assimilation solver
# =====================================================================
def run_assimilation(K, M, F_noisy_vec, Vr, H, y_obs,
                     tau, kappa, sigma2_me, sigma2_y, Sigma_alpha_inv):
    n_dof = K.shape[0]
    N_obs = H.shape[0]

    Q = csc_matrix((1.0 / tau) * (kappa**2 * M + K))
    R = identity(n_dof, format='csc') / sigma2_me
    Gamma_inv = csc_matrix((1.0 / sigma2_y) * np.eye(N_obs))
    HtGiH = H.T @ Gamma_inv @ H

    KtRK = K.T @ R @ K
    Lambda_11 = Q + KtRK + HtGiH

    A_22 = np.asarray(Vr.T @ (HtGiH @ Vr))
    Lambda_22 = A_22 + np.asarray(Sigma_alpha_inv)

    Lambda_12 = csc_matrix(HtGiH @ Vr)
    Lambda_21 = Lambda_12.T
    Lambda_22_csc = csc_matrix(Lambda_22)

    Lambda = bmat([
        [Lambda_11, Lambda_12],
        [Lambda_21, Lambda_22_csc]
    ], format='csc')

    HtGi_y = np.asarray(H.T @ (Gamma_inv @ y_obs)).flatten()
    KtR_F  = np.asarray(K.T @ (R @ F_noisy_vec)).flatten()
    b1 = HtGi_y + KtR_F
    b2 = np.asarray(Vr.T @ HtGi_y).flatten()
    b = np.concatenate([b1, b2])

    sol = spsolve(Lambda, b)
    u_post = sol[:n_dof]
    return u_post

# =====================================================================
# Main loop: testing different R_FIXED
# =====================================================================
print("\n" + "=" * 60)
print(f"Starting batch test of R_VALUES ({len(R_VALUES)} groups)")
print("=" * 60)

results_summary = []

for r_val in R_VALUES:
    if r_val > len(eigvals_nz_all):
        print(f"\n[SKIP] R_FIXED={r_val}: insufficient nonzero eigenvalues ({len(eigvals_nz_all)})")
        continue
        
    print(f"\n--- Processing R_FIXED = {r_val} ---")
    
    # 1. Slice to get basis Vr and eigenvalues for current R
    eigvals_r = eigvals_nz_all[:r_val]
    Vr        = Vr_nz_all[:, :r_val]
    
    # 2. Analytic hyperparameter estimation
    params_analytical = estimate_hyperparams(
        K, M, u_true_nodes, F_noisy_vec, C_TRUTH, Vr, eigvals_r
    )
    
    # 3. Run analytic assimilation
    u_post_analytical = run_assimilation(
        K, M, F_noisy_vec, Vr, H, u_obs,
        tau=params_analytical['tau'],
        kappa=params_analytical['kappa'],
        sigma2_me=params_analytical['sigma2_me'],
        sigma2_y=params_analytical['sigma2_y'],
        Sigma_alpha_inv=params_analytical['Sigma_alpha_inv']
    )
    L2_err_analytical, _ = l2_error(u_post_analytical, u_true_nodes, M)
    
    # 4. Optuna optimization
    def objective(trial):
        tau = trial.suggest_float('tau', 1e-1, 1e4, log=True)
        kappa = trial.suggest_float('kappa', 1e-3, 5.0, log=True)
        sigma2_me = trial.suggest_float('sigma2_me', 1e-2, 1e3, log=True)
        alpha_pen = trial.suggest_float('alpha_pen', 1e-6, 1e2, log=True)
        
        Sigma_alpha_inv = np.eye(r_val) * alpha_pen
        
        u_post = run_assimilation(
            K, M, F_noisy_vec, Vr, H, u_obs,
            tau=tau, kappa=kappa, sigma2_me=sigma2_me,
            sigma2_y=sigma2_y_fixed,
            Sigma_alpha_inv=Sigma_alpha_inv
        )
        L2_err, _ = l2_error(u_post, u_true_nodes, M)
        return L2_err

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    t_opt = time.time() - t0
    
    best = study.best_params
    
    # 5. Run Optuna best assimilation
    Sigma_alpha_inv_best = np.eye(r_val) * best['alpha_pen']
    u_post_optuna = run_assimilation(
        K, M, F_noisy_vec, Vr, H, u_obs,
        tau=best['tau'], kappa=best['kappa'],
        sigma2_me=best['sigma2_me'],
        sigma2_y=sigma2_y_fixed,
        Sigma_alpha_inv=Sigma_alpha_inv_best
    )
    L2_err_optuna, _ = l2_error(u_post_optuna, u_true_nodes, M)
    
    # 6. Record results
    results_summary.append({
        'R': r_val,
        'L2_Base': L2_err_base,
        'L2_Analytical': L2_err_analytical,
        'L2_Optuna': L2_err_optuna,
        'Optuna_Time': t_opt,
        'Params_Ana': params_analytical,
        'Params_Opt': best
    })
    
    print(f"  Done | Analytical L2: {L2_err_analytical:.2e} | Optuna L2: {L2_err_optuna:.2e} | Time: {t_opt:.1f}s")

# =====================================================================
# Step 8: Comprehensive results
# =====================================================================
print("\n" + "=" * 90)
print("=" * 90)

header = f"{'R':<6} | {'L2 Error (Base)':<18} | {'L2 Error (Ana)':<18} | {'L2 Error (Opt)':<18} | {'Opt Time(s)':<10}"
print(header)
print("-" * 90)

for res in results_summary:
    print(f"{res['R']:<6} | {res['L2_Base']:<18.6e} | {res['L2_Analytical']:<18.6e} | {res['L2_Optuna']:<18.6e} | {res['Optuna_Time']:<10.1f}")

print("\n" + "=" * 90)
print("Detailed parameter comparison (sampled)")
print("=" * 90)
sample_indices = [0, len(results_summary)//2, len(results_summary)-1]
for idx in sample_indices:
    if idx < len(results_summary):
        res = results_summary[idx]
        print(f"\n--- R = {res['R']} ---")
        p_ana = res['Params_Ana']
        p_opt = res['Params_Opt']
        print(f"  Analytical: tau={p_ana['tau']:.2e}, kappa={p_ana['kappa']:.4f}, sigma_me={p_ana['sigma2_me']:.2e}")
        print(f"  Optuna:     tau={p_opt['tau']:.2e}, kappa={p_opt['kappa']:.4f}, sigma_me={p_opt['sigma2_me']:.2e}, alpha={p_opt['alpha_pen']:.2e}")

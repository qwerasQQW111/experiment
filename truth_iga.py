"""
statFEM_NURBS_batch.py — NURBS IGA data assimilation (single-patch open surface, Robin BC, physical heat source)
=====================================================================
1. Load real physics field (Robin thermal BC + 3D Gaussian volumetric heat source)
2. Analytic hyperparameter estimation
3. Optuna Bayesian optimization
4. Compare assimilation performance under different R_FIXED
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import csc_matrix, lil_matrix, identity, bmat
from scipy.interpolate import RegularGridInterpolator
import warnings, sys, time
from pathlib import Path

warnings.filterwarnings('ignore')


from surface_model import (
    load_nurbs_from_npz, refine_nurbs_h, refine_nurbs_p, _eval_nurbs_surface, 
    Surface_NURBS, precompute_iga_full, compute_iga_l2_from_precomp, compute_error_field_as_nurbs_surface,
    visualize_error_field_2d
)
from surface_fem_solver import (
    assemble_nurbs_matrices, assemble_nurbs_load, _nurbs_eval,l2_error_nurbs
)
from pullback_mms import evaluate_nurbs_rational_derivs, corrupt_source_term

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# Parameter settings (from phys.py physics constraints)
# =====================================================================
NURBS_NPZ = "model/169.npz"
N_GRID    = 800

# Physical constants
K_COND    = 160.0            # Thermal conductivity
DELTA     = 0.005            # Thickness
H_SURF    = 200.0            # Surface heat transfer coefficient
H_EDGE    = 500.0            # Edge heat transfer coefficient
U_INF     = 0.0              # Ambient temperature

# PDE coefficients (c = 2h / (k*delta), beta = h_edge / (k*delta))
C_TRUTH   = (2.0 * H_SURF) / (K_COND * DELTA)   # = 500.0
BETA_VAL  = H_EDGE / (K_COND * DELTA)           # = 625.0
C_MODEL   = 200.0              # Assumed model reaction coefficient (simulating model misspecification)

N_SENSORS = 16               # Number of observations
NOISE_STD = 0.5              # Observation noise (very low)

WHITE_STD = 0.001            # Source term white noise strength
LF_AMP    = 0.05             # Source term bias amplitude
LF_MODES  = 2                # Source term bias frequency count

R_VALUES  = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
N_TRIALS  = 100              # Optuna optimization trials

NQ        = 5
SEED      = 42
np.random.seed(SEED)


# =====================================================================
# Helper functions: Load NURBS & build operators
# =====================================================================
def caculate_spans(nurbs):
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

    return elem_spans

def load_nurbs(npz_path, target_u=None, target_v=None):
    """
    Load and optionally refine a NURBS surface.
    
    Args:
        npz_path: path to .npz file
        target_u: target refined control points in U direction (optional; skip if None or <= current)
        target_v: target refined control points in V direction (optional; skip if None or <= current)
    """
    nurbs = load_nurbs_from_npz(str(npz_path))
    
    # Determine whether refinement is needed
    u_req = target_u if target_u is not None else nurbs.ctrl_u
    v_req = target_v if target_v is not None else nurbs.ctrl_v
    
    # Only trigger refinement when target > current
    if u_req > nurbs.ctrl_u or v_req > nurbs.ctrl_v:

        nurbs, a_, b_ = refine_nurbs_h(nurbs, max(u_req, nurbs.ctrl_u), max(v_req, nurbs.ctrl_v))

    n_ctrl_u = nurbs.ctrl_u
    n_ctrl_v = nurbs.ctrl_v
    elem_spans = caculate_spans(nurbs)

            
    return nurbs, elem_spans, n_ctrl_u, n_ctrl_v
# Refine degree first, then control points, to avoid K/M matrix explosion
# Main issue is L2 error
def truth_nurbs(npz_path, target_u_deg=None, target_v_deg=None, target_u_ctrl=None,target_v_ctrl=None):

    nurbs = load_nurbs_from_npz(str(npz_path))

    # Determine whether refinement is needed
    u_deg_req = target_u_deg if target_u_deg is not None else nurbs.degree_u
    v_deg_req = target_v_deg if target_v_deg is not None else nurbs.degree_v
    
    # Only trigger refinement when target > current
    if u_deg_req > nurbs.degree_u or v_deg_req > nurbs.degree_v:

        nurbs_deg, _, _ = refine_nurbs_p(nurbs, max(u_deg_req, nurbs.degree_u), max(v_deg_req, nurbs.degree_v))
    else:
        nurbs_deg = nurbs

    u_ctrl_req = target_u_ctrl if target_u_ctrl is not None else nurbs.ctrl_u
    v_ctrl_req = target_v_ctrl if target_v_ctrl is not None else nurbs.ctrl_v
    
    # Only trigger refinement when target > current
    if u_ctrl_req > nurbs.ctrl_u or v_ctrl_req > nurbs.ctrl_v:
        nurbs_ctrl, _, _ = refine_nurbs_h(nurbs_deg, max(u_ctrl_req, nurbs.ctrl_u), max(v_ctrl_req, nurbs.ctrl_v))
    else:
        nurbs_ctrl = nurbs_deg
    

    n_ctrl_u = nurbs_ctrl.ctrl_u
    n_ctrl_v = nurbs_ctrl.ctrl_v
    elem_spans = caculate_spans(nurbs_ctrl)
    print(nurbs_ctrl.ctrl_u,nurbs_ctrl.ctrl_v,nurbs_ctrl.degree_u,nurbs_ctrl.degree_v)

            
    return nurbs_ctrl, elem_spans, n_ctrl_u, n_ctrl_v

def build_observation_operator(nurbs, uv_sensors):
    n_obs = uv_sensors.shape[0]
    n_dof = nurbs.ctrl_u * nurbs.ctrl_v
    U, V = nurbs.knotvector_u, nurbs.knotvector_v
    p, q = nurbs.degree_u, nurbs.degree_v
    n_ctrl_u, n_ctrl_v = nurbs.ctrl_u, nurbs.ctrl_v
    H_lil = lil_matrix((n_obs, n_dof))
    for i in range(n_obs):
        u, v = uv_sensors[i]
        su = np.searchsorted(U, u, side='right') - 1
        su = np.clip(su, p, n_ctrl_u - 1)
        sv = np.searchsorted(V, v, side='right') - 1
        sv = np.clip(sv, q, n_ctrl_v - 1)
        R, _, _, ids = _nurbs_eval(u, v, su, sv, nurbs)
        for k in range(len(R)):
            iu, iv = ids[k]
            dof = iu * n_ctrl_v + iv
            H_lil[i, dof] = R[k]
    return H_lil.tocsr()

def assemble_nurbs_robin_bc(nurbs, beta_coeff, u_inf, nq=5):
    """General Robin BC matrix and load assembly on NURBS parametric domain boundary"""
    n_dof = nurbs.ctrl_u * nurbs.ctrl_v
    K_robin = lil_matrix((n_dof, n_dof))
    F_robin = np.zeros(n_dof)
    
    from numpy.polynomial.legendre import leggauss
    xi_1d, w_1d = leggauss(nq)
    
    unique_u = np.unique(nurbs.knotvector_u)
    unique_v = np.unique(nurbs.knotvector_v)
    
    def integrate_edge(const_u, const_v, var_range, is_u_var):
        ua, ub = var_range
        for k in range(nq):
            t = ua + 0.5 * (xi_1d[k] + 1) * (ub - ua)
            dt = 0.5 * (ub - ua)
            w = w_1d[k]
            
            u = t if is_u_var else const_u
            v = const_v if is_u_var else t
            
            # Boundary handling to prevent overflow
            u_eval = min(max(u, 1e-12), 1.0 - 1e-12)
            v_eval = min(max(v, 1e-12), 1.0 - 1e-12)
            
            su = np.searchsorted(nurbs.knotvector_u, u_eval, side='right') - 1
            su = np.clip(su, nurbs.degree_u, nurbs.ctrl_u - 1)
            sv = np.searchsorted(nurbs.knotvector_v, v_eval, side='right') - 1
            sv = np.clip(sv, nurbs.degree_v, nurbs.ctrl_v - 1)
            
            R, dRu, dRv, ids = _nurbs_eval(u_eval, v_eval, su, sv, nurbs)
            X = nurbs.control_points[ids[:, 0], ids[:, 1]]
            
            # Compute tangent for arc-length element ds
            tangent = np.sum(X * dRu[:, None], axis=0) if is_u_var else np.sum(X * dRv[:, None], axis=0)
            ds = np.linalg.norm(tangent) * dt * w
            
            dof = ids[:, 0] * nurbs.ctrl_v + ids[:, 1]
            for i in range(len(dof)):
                F_robin[dof[i]] += beta_coeff * u_inf * R[i] * ds
                for j in range(len(dof)):
                    K_robin[dof[i], dof[j]] += beta_coeff * R[i] * R[j] * ds

    # Integrate over four edges
    for i in range(len(unique_u)-1):
        if unique_u[i+1] - unique_u[i] > 1e-10:
            integrate_edge(None, 0.0, (unique_u[i], unique_u[i+1]), True)
            integrate_edge(None, 1.0, (unique_u[i], unique_u[i+1]), True)
            
    for j in range(len(unique_v)-1):
        if unique_v[j+1] - unique_v[j] > 1e-10:
            integrate_edge(0.0, None, (unique_v[j], unique_v[j+1]), False)
            integrate_edge(1.0, None, (unique_v[j], unique_v[j+1]), False)
            
    return K_robin.tocsc(), F_robin

def get_sensors_data(nurbs, u_true_vec, n_sensors, noise_std, rng_seed):
    """Generate sensor observations directly from truth coefficient vector"""
    rng = np.random.default_rng(rng_seed)
    from scipy.stats import qmc
    sampler = qmc.Halton(d=2, seed=rng_seed)
    uv_sensors = sampler.random(n_sensors)
    pad = 1e-6
    uv_sensors = pad + uv_sensors * (1.0 - 2.0 * pad)
    H = build_observation_operator(nurbs, uv_sensors)
    u_true_sens = H @ u_true_vec
    u_obs = u_true_sens + rng.normal(0.0, noise_std, size=n_sensors)
    return u_obs, H, uv_sensors


print("=" * 60)
print("Step 1-5: Geometry loading and physics field assembly (Robin BC + 3D heat source)")
print("=" * 60)

# --- 1. Load NURBS & basic discretization ---
nurbs, elem_spans, n_ctrl_u, n_ctrl_v = load_nurbs(NURBS_NPZ,33,33)
forward_nurbs, elem_spans_forward, _, _ = truth_nurbs(NURBS_NPZ,6,6,50,50)

n_dof = n_ctrl_u * n_ctrl_v
print(f"[IGA] DOFs={n_dof}, ctrl=({n_ctrl_u},{n_ctrl_v}), deg=({nurbs.degree_u},{nurbs.degree_v})")

K, M = assemble_nurbs_matrices(nurbs, elem_spans, nq=NQ)
K_robin, F_robin = assemble_nurbs_robin_bc(nurbs, BETA_VAL, U_INF, nq=NQ)

# Under Robin BC, no fixed Dirichlet nodes; all DOFs are free.
interior_dofs = np.arange(n_dof)
n_int = n_dof
print(f"[IGA] Robin BC: all DOFs free, interior DOFs = {n_int}")

# --- 2. Physics source term generation (inherited from phys.py) ---
nodes = forward_nurbs.control_points.reshape(-1, 3)
cx, cy, cz = nodes.mean(axis=0)
diag = np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))
radius = diag * 0.08
print(f"  [Physics] Source center = ({cx:.2f}, {cy:.2f}, {cz:.2f}), Gaussian radius = {radius:.3f}")

def source_func_3d(x, y, z):
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    q_max = 800000.0
    q_vol = q_max * np.exp(-dist_sq / (2 * radius**2))
    return q_vol / K_COND

pad = 1e-6
u_vals = np.linspace(pad, 1.0 - pad, N_GRID)
v_vals = np.linspace(pad, 1.0 - pad, N_GRID)
U_grid, V_grid = np.meshgrid(u_vals, v_vals, indexing='ij')
uv_pts = np.stack([U_grid.ravel(), V_grid.ravel()], axis=-1)

# Generate noise-free parametric domain source term grid
derivs = evaluate_nurbs_rational_derivs(uv_pts, forward_nurbs.control_points, forward_nurbs.weights,
                                        forward_nurbs.knotvector_u, forward_nurbs.knotvector_v, forward_nurbs.degree_u, forward_nurbs.degree_v)
coords = derivs['S']
f_clean_flat = np.array([source_func_3d(x,y,z) for x,y,z in coords])
f_clean_grid = f_clean_flat.reshape((N_GRID, N_GRID))

# Apply noise to simulate real-world degradation
f_noisy_grid, _, _ = corrupt_source_term(
    f_clean_grid, N_GRID, white_std=WHITE_STD, low_freq_amplitude=LF_AMP,
    low_freq_modes=LF_MODES, rng=SEED
)

f_clean_interp = RegularGridInterpolator((u_vals, v_vals), f_clean_grid, method='cubic', bounds_error=False, fill_value=None)
f_noisy_interp = RegularGridInterpolator((u_vals, v_vals), f_noisy_grid, method='cubic', bounds_error=False, fill_value=None)

# --- 3. Solve truth physics field and baseline (prior) model field ---
#F_clean_vol = assemble_nurbs_load(nurbs, elem_spans, f_clean_interp, nq=NQ)
F_noisy_vol = assemble_nurbs_load(nurbs, elem_spans, f_noisy_interp, nq=NQ)

# Truth solution (true reaction coefficient + true source + Robin BC)


F_clean_vol_forward = assemble_nurbs_load(forward_nurbs, elem_spans_forward, f_clean_interp, nq=NQ)

K_forward, M_forward = assemble_nurbs_matrices(forward_nurbs, elem_spans_forward, nq=NQ)
K_robin_forward, F_robin_forward = assemble_nurbs_robin_bc(forward_nurbs, BETA_VAL, U_INF, nq=NQ)

K_true_sys = K_forward + C_TRUTH * M_forward + K_robin_forward
F_true_sys = F_clean_vol_forward + F_robin_forward
# print(np.linalg.cond(K_true_sys),np.linalg.cond(F_true_sys))


u_true_full = spsolve(K_true_sys, F_true_sys)
u_true_int = u_true_full



u_true_3d = u_true_full.reshape(forward_nurbs.ctrl_u, forward_nurbs.ctrl_v, 1)

# 2. Densely sample true physics field values on parameter domain grid (uv_pts)
u_true_eval = _eval_nurbs_surface(
    uv_pts, u_true_3d, forward_nurbs.weights,
    forward_nurbs.knotvector_u, forward_nurbs.knotvector_v,
    forward_nurbs.degree_u, forward_nurbs.degree_v
)
u_true_grid = u_true_eval[:, 0].reshape(N_GRID, N_GRID)

# 3. Build scipy interpolator (for fast numerical integration)
u_true_interp = RegularGridInterpolator(
    (u_vals, v_vals), u_true_grid, 
    method='cubic', bounds_error=False, fill_value=None
)

# 4. Assemble load vector on coarse grid (nurbs) F_proj = \int u_true * R_i d\Omega
F_u_true_proj = assemble_nurbs_load(nurbs, elem_spans, u_true_interp, nq=NQ)

# 5. Solve mass matrix equation M * u_proj = F_proj to get perfectly aligned coarse-grid coefficients!
u_true_projected = spsolve(M, F_u_true_proj)





# Baseline model (degraded reaction coefficient + noisy source + Robin BC)
K_model_sys = K + C_MODEL * M + K_robin
F_noisy_sys = F_noisy_vol + F_robin
u_base_full = spsolve(K_model_sys, F_noisy_sys)
u_base_int = u_base_full

# --- 4. Generate sensor data and initialize assimilation operators ---
u_obs, H_full, uv_sensors = get_sensors_data(nurbs, u_true_projected, N_SENSORS, NOISE_STD, SEED)
H_int = H_full
sigma2_y_fixed = NOISE_STD ** 2

K_int = K_model_sys.tocsc()
M_int = M.tocsc()
F_noisy_int = F_noisy_sys

precomp = precompute_iga_full(forward_nurbs, u_true_int, nurbs, NQ)

# --- Helper functions ---
def weighted_mean(vec, M_mat):
    return float(np.dot(M_mat @ vec, np.ones(len(vec)))) / float(np.sum(M_mat))

def weighted_var(vec, M_mat):
    m = weighted_mean(vec, M_mat)
    return weighted_mean((vec - m)**2, M_mat)


# L2_err_base = compute_iga_field_l2_error(
#     nurbs1=forward_nurbs, u_vec1=u_true_int,
#     nurbs2=nurbs, u_vec2=u_base_int,
#     n_gauss=NQ
# )

L2_err_base = compute_iga_l2_from_precomp(u_base_int,precomp)

L2_true = np.sqrt(u_true_full @ (M_forward @ u_true_full))
print(f"[Baseline] u_base vs u_true: L2 = {L2_err_base:.6e}, |u_true| = {L2_true:.6e}")

# --- 5. Precompute eigenvalues and eigenvectors at max scale ---
MAX_R = max(R_VALUES)
print(f"  Precomputing eigenvalues (Max R = {MAX_R})...")
N_EIG_REQUEST = min(MAX_R + 2, n_int - 1)

try:
    eigvals_all, Vr_all = eigsh(K_int, k=N_EIG_REQUEST, M=M_int, sigma=0.0, which='LM')
except Exception as e:
    print(f"  [WARNING] eigenvalue computation failed: {e}. try reducing requested dimension...")
    N_EIG_REQUEST = min(5, n_int - 1)
    eigvals_all, Vr_all = eigsh(K_int, k=N_EIG_REQUEST, M=M_int, sigma=0.0, which='LM')

order = np.argsort(eigvals_all)
eigvals_all = eigvals_all[order]
Vr_all = Vr_all[:, order]

tol_eig = 1e-12
mask_nz = np.abs(eigvals_all) > tol_eig
eigvals_nz_all = eigvals_all[mask_nz]
Vr_nz_all = Vr_all[:, mask_nz]

print(f"  Precomputation done. Found {len(eigvals_nz_all)}  nonzero eigenvalues.")


# =====================================================================
# Step 6: Analytic hyperparameter estimation
# =====================================================================



def estimate_hyperparams(K_int, M_int, u_true_int, F_noisy_int, Vr, eigvals_r):

    sigma2_y = sigma2_y_fixed

    res = F_noisy_int - K_int @ u_true_int
    sigma2_me = weighted_mean(res**2, M_int)
    var_u = weighted_var(u_true_int, M_int)
    lam1 = eigvals_r[0]
    kappa2 = max(0.5 * lam1, 1e-4)
    kappa = np.sqrt(kappa2)
    tau = var_u * (kappa2 + lam1)
    u_true_anom = u_true_int - weighted_mean(u_true_int, M_int)
    alpha_true = Vr.T @ (M_int @ u_true_anom)
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
def run_assimilation(K_int, M_int, F_noisy_int, Vr, H_int, y_obs,
                     tau, kappa, sigma2_me, sigma2_y, Sigma_alpha_inv):
    n_int = K_int.shape[0]
    N_obs = H_int.shape[0]
    Q = csc_matrix((1.0 / tau) * (kappa**2 * M_int + K_int))
    R_mat = identity(n_int, format='csc') / sigma2_me
    Gamma_inv = csc_matrix((1.0 / sigma2_y) * np.eye(N_obs))
    HtGiH = H_int.T @ Gamma_inv @ H_int
    KtRK = K_int.T @ R_mat @ K_int
    Lambda_11 = Q + KtRK + HtGiH
    A_22 = np.asarray(Vr.T @ (HtGiH @ Vr))
    Lambda_22 = A_22 + np.asarray(Sigma_alpha_inv)
    Lambda_12 = csc_matrix(HtGiH @ Vr)
    Lambda_21 = Lambda_12.T
    Lambda = bmat([[Lambda_11, Lambda_12], [Lambda_21, csc_matrix(Lambda_22)]], format='csc')
    HtGi_y = np.asarray(H_int.T @ (Gamma_inv @ y_obs)).flatten()
    KtR_F = np.asarray(K_int.T @ (R_mat @ F_noisy_int)).flatten()
    b1 = HtGi_y + KtR_F
    b2 = np.asarray(Vr.T @ HtGi_y).flatten()
    sol = spsolve(Lambda, np.concatenate([b1, b2]))
    return sol[:n_int], sol[n_int:], np.linalg.inv(Lambda_22)

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
    
    # 1. Slice eigenbasis for current R
    eigvals_r = eigvals_nz_all[:r_val]
    Vr        = Vr_nz_all[:, :r_val]
    
    # 2. Analytic hyperparameter estimation
    params_analytical = estimate_hyperparams(
        K_int, M_int, u_true_projected, F_noisy_int, Vr, eigvals_r
    )
    
    # 3. Run analytical assimilation
    u_post_ana_int, _, _ = run_assimilation(
        K_int, M_int, F_noisy_int, Vr, H_int, u_obs,
        tau=params_analytical['tau'], kappa=params_analytical['kappa'],
        sigma2_me=params_analytical['sigma2_me'],
        sigma2_y=params_analytical['sigma2_y'],
        Sigma_alpha_inv=params_analytical['Sigma_alpha_inv']
    )
    # L2_err_analytical = compute_iga_field_l2_error(
    #     nurbs1=forward_nurbs, u_vec1=u_true_int,
    #     nurbs2=nurbs, u_vec2=u_post_ana_int,
    #     n_gauss=NQ
    # )
    L2_err_analytical = compute_iga_l2_from_precomp(u_post_ana_int,precomp)
    
    # 4. Optuna optimization
    def objective(trial):
        tau = trial.suggest_float('tau', 1e-1, 1e4, log=True)
        kappa = trial.suggest_float('kappa', 1e-3, 5.0, log=True)
        sigma2_me = trial.suggest_float('sigma2_me', 1e-2, 1e3, log=True)
        alpha_pen = trial.suggest_float('alpha_pen', 1e-6, 1e2, log=True)
        
        Sigma_alpha_inv = np.eye(r_val) * alpha_pen
        
        u_post_int, _, _ = run_assimilation(
            K_int, M_int, F_noisy_int, Vr, H_int, u_obs,
            tau=tau, kappa=kappa, sigma2_me=sigma2_me,
            sigma2_y=sigma2_y_fixed, Sigma_alpha_inv=Sigma_alpha_inv
        )
        # return compute_iga_field_l2_error(
        #     nurbs1=forward_nurbs, u_vec1=u_true_int,
        #     nurbs2=nurbs, u_vec2=u_post_int,
        #     n_gauss=NQ
        # )

        return compute_iga_l2_from_precomp(u_post_int,precomp)

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    t_opt = time.time() - t0
    
    best = study.best_params
    
    # 5. Run best Optuna assimilation
    Sigma_alpha_inv_best = np.eye(r_val) * best['alpha_pen']
    u_post_opt_int, _, _ = run_assimilation(
        K_int, M_int, F_noisy_int, Vr, H_int, u_obs,
        tau=best['tau'], kappa=best['kappa'],
        sigma2_me=best['sigma2_me'],
        sigma2_y=sigma2_y_fixed, Sigma_alpha_inv=Sigma_alpha_inv_best
    )


    # err_field = compute_error_field_as_nurbs_surface(nurbs,u_post_opt_int,forward_nurbs,u_true_int,NQ)
    # visualize_error_field_2d(err_field)
    L2_err_optuna = compute_iga_l2_from_precomp(u_post_opt_int,precomp)
    
    
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

"""
statFEM_P2_Robin_CrossSpace.py — High-precision P2 data assimilation based on cross-space continuous L2 error
===================================================================================
1. Truth field: generated from high-fidelity high-order NURBS model (3D Gaussian heat source + Robin thermal BC).
2. Assimilation field: built on low-order P2 finite element space (missing reaction coefficient + noisy source).
3. Observations & hyperparameter estimation: computed via P2 space nodal projection.
4. Error metric (core modification): implements cross-space continuous L2 integration, independent of any space projection approximation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import csc_matrix, lil_matrix, identity, bmat
from scipy.interpolate import RegularGridInterpolator
import warnings, time

warnings.filterwarnings('ignore')

from surface_model import (
    load_nurbs_from_npz, refine_nurbs_h, refine_nurbs_p, _eval_nurbs_surface,
    load_p2_from_npz, Surface_NURBS, Surface_P2,
    precompute_l2_error_data,compute_l2_error_from_precomputed,
    compute_error_field_as_p2_surface,
    visualize_error_field_2d
)
from surface_fem_solver import (
    assemble_nurbs_matrices, assemble_nurbs_load, _nurbs_eval, assemble_p2_matrices
)
from pullback_mms import evaluate_nurbs_rational_derivs, corrupt_source_term

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# Parameter settings (synced from NURBS_statFEM physics benchmark)
# =====================================================================
P2_NPZ      = "85_P2.npz"    # P2 finite element model mesh file
NURBS_NPZ   = "85.npz"             # High-fidelity geometry reference file
N_GRID      = 800                   # High-resolution source term and physics field analysis grid

# Physical constant settings
K_COND      = 160.0                 # Thermal conductivity
DELTA       = 0.005                 # Thin plate structure thickness
H_SURF      = 200.0                 # Surface convective heat transfer coefficient
H_EDGE      = 500.0                 # Edge convective heat transfer coefficient
U_INF       = 0.0                   # Ambient reference temperature

C_TRUTH     = (2.0 * H_SURF) / (K_COND * DELTA)
BETA_VAL    = H_EDGE / (K_COND * DELTA)
C_MODEL     = 200.0                   # Assume degraded Base model entirely missing reaction coefficient

N_SENSORS   = 4                    # Number of spatially random sensor observation points
NOISE_STD   = 0.5                   # Sensor measurement Gaussian noise std dev

WHITE_STD   = 0.001                 # Source term spatial white noise
LF_AMP      = 0.05                  # Source term low-frequency spatial random bias amplitude
LF_MODES    = 2                     # Number of low-frequency perturbation modes

R_VALUES    = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
N_TRIALS    = 100                   # Optuna optimization iterations
NQ          = 5                     # High-precision Gaussian quadrature points
SEED        = 42
np.random.seed(SEED)





# =====================================================================
# Auxiliary assembly module (supports P2 Robin BC operator and NURBS element span location)
# =====================================================================
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


def caculate_spans(nurbs):
    n_ctrl_u, n_ctrl_v = nurbs.ctrl_u, nurbs.ctrl_v
    deg_u, deg_v = nurbs.degree_u, nurbs.degree_v
    U, V = nurbs.knotvector_u, nurbs.knotvector_v
    unique_u, unique_v = np.unique(U), np.unique(V)
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


def assemble_nurbs_robin_bc(nurbs, beta_coeff, u_inf, nq=5):
    n_dof = nurbs.ctrl_u * nurbs.ctrl_v
    K_robin = lil_matrix((n_dof, n_dof))
    F_robin = np.zeros(n_dof)
    from numpy.polynomial.legendre import leggauss
    xi_1d, w_1d = leggauss(nq)
    unique_u, unique_v = np.unique(nurbs.knotvector_u), np.unique(nurbs.knotvector_v)
    
    def integrate_edge(const_u, const_v, var_range, is_u_var):
        ua, ub = var_range
        for k in range(nq):
            t = ua + 0.5 * (xi_1d[k] + 1) * (ub - ua)
            dt = 0.5 * (ub - ua)
            u = t if is_u_var else const_u
            v = const_v if is_u_var else t
            u_eval = min(max(u, 1e-12), 1.0 - 1e-12)
            v_eval = min(max(v, 1e-12), 1.0 - 1e-12)
            su = np.clip(np.searchsorted(nurbs.knotvector_u, u_eval, side='right') - 1, nurbs.degree_u, nurbs.ctrl_u - 1)
            sv = np.clip(np.searchsorted(nurbs.knotvector_v, v_eval, side='right') - 1, nurbs.degree_v, nurbs.ctrl_v - 1)
            R, dRu, dRv, ids = _nurbs_eval(u_eval, v_eval, su, sv, nurbs)
            X = nurbs.control_points[ids[:, 0], ids[:, 1]]
            tangent = np.sum(X * dRu[:, None], axis=0) if is_u_var else np.sum(X * dRv[:, None], axis=0)
            ds = np.linalg.norm(tangent) * dt * w_1d[k]
            dof = ids[:, 0] * nurbs.ctrl_v + ids[:, 1]
            for i in range(len(dof)):
                F_robin[dof[i]] += beta_coeff * u_inf * R[i] * ds
                for j in range(len(dof)):
                    K_robin[dof[i], dof[j]] += beta_coeff * R[i] * R[j] * ds

    for i in range(len(unique_u)-1):
        if unique_u[i+1] - unique_u[i] > 1e-10:
            integrate_edge(None, 0.0, (unique_u[i], unique_u[i+1]), True)
            integrate_edge(None, 1.0, (unique_u[i], unique_u[i+1]), True)
    for j in range(len(unique_v)-1):
        if unique_v[j+1] - unique_v[j] > 1e-10:
            integrate_edge(0.0, None, (unique_v[j], unique_v[j+1]), False)
            integrate_edge(1.0, None, (unique_v[j], unique_v[j+1]), False)
    return K_robin.tocsc(), F_robin

def assemble_p2_robin_bc(p2: Surface_P2, beta_coeff, u_inf, nq=3):
    n_nodes = p2.nodes_2d.shape[0]
    K_robin = lil_matrix((n_nodes, n_nodes))
    F_robin = np.zeros(n_nodes)
    from numpy.polynomial.legendre import leggauss
    xi, w = leggauss(nq)
    t_gauss, w_gauss = 0.5 * (xi + 1.0), 0.5 * w

    def N_1d(t): return np.array([2*t**2 - 3*t + 1, -4*t**2 + 4*t, 2*t**2 - t])
    def dN_1d(t): return np.array([4*t - 3, -8*t + 4, 4*t - 1])

    n_u, n_v = p2.n_u, p2.n_v
    n_vert = (n_u + 1) * (n_v + 1)
    n_h = n_u * (n_v + 1)

    def add_edge(idx0, idx1, idx2):
        nodes = [idx0, idx1, idx2]
        X = p2.nodes_3d[nodes]
        for t, weight in zip(t_gauss, w_gauss):
            N_val = N_1d(t)
            dX_dt = dN_1d(t) @ X
            ds = np.linalg.norm(dX_dt)
            for a in range(3):
                F_robin[nodes[a]] += beta_coeff * u_inf * N_val[a] * ds * weight
                for b in range(3):
                    K_robin[nodes[a], nodes[b]] += beta_coeff * N_val[a] * N_val[b] * ds * weight

    for i in range(n_u):
        add_edge(i*(n_v+1), n_vert + i*(n_v+1), (i+1)*(n_v+1))
        add_edge(n_v + i*(n_v+1), n_vert + n_v + i*(n_v+1), n_v + (i+1)*(n_v+1))
    for j in range(n_v):
        add_edge(j, n_vert + n_h + j, j+1)
        add_edge(j + n_u*(n_v+1), n_vert + n_h + j + n_u*n_v, j + 1 + n_u*(n_v+1))
    return K_robin.tocsc(), F_robin

def build_observation_operator(p2, uv_sensors):
    n_obs = uv_sensors.shape[0]
    n_u, n_v = p2.n_u, p2.n_v
    du, dv = 1.0 / n_u, 1.0 / n_v
    H_lil = lil_matrix((n_obs, p2.nodes_2d.shape[0]))

    n_vert = (n_u + 1) * (n_v + 1)
    n_h = n_u * (n_v + 1)
    nve = (n_u + 1) * n_v

    for row, (u_raw, v_raw) in enumerate(uv_sensors):
        u = float(np.clip(u_raw, 0.0, 1.0))
        v = float(np.clip(v_raw, 0.0, 1.0))
        ci = int(np.clip(u / du, 0, n_u - 1))
        cj = int(np.clip(v / dv, 0, n_v - 1))
        xi = (u - ci * du) / du
        eta = (v - cj * dv) / dv

        i00 = cj + ci * (n_v + 1)
        i10 = cj + (ci + 1) * (n_v + 1)
        i11 = (cj + 1) + (ci + 1) * (n_v + 1)
        i01 = (cj + 1) + ci * (n_v + 1)
        ih00 = n_vert + cj + ci * (n_v + 1)
        ih01 = n_vert + (cj + 1) + ci * (n_v + 1)
        iv10 = n_vert + n_h + cj + (ci + 1) * n_v
        iv00 = n_vert + n_h + cj + ci * n_v
        id00 = n_vert + n_h + nve + cj + ci * n_v

        if xi >= eta:
            la, lb, lc = 1.0 - xi, xi - eta, eta
            idx = (i00, i10, i11, ih00, iv10, id00)
        else:
            la, lb, lc = 1.0 - eta, xi, eta - xi
            idx = (i00, i11, i01, id00, ih01, iv00)

        vals = (
            la * (2.0 * la - 1.0),
            lb * (2.0 * lb - 1.0),
            lc * (2.0 * lc - 1.0),
            4.0 * la * lb,
            4.0 * lb * lc,
            4.0 * la * lc,
        )
        for col, val in zip(idx, vals):
            H_lil[row, int(col)] = float(val)

    return H_lil.tocsr()


def make_sensor_locations(n_sensors, rng_seed, pad=1e-6):
    """Generate the shared sensor coordinates used by all discretizations."""
    from scipy.stats import qmc

    sampler = qmc.Halton(d=2, seed=rng_seed)
    uv_sensors = sampler.random(n_sensors)
    return pad + uv_sensors * (1.0 - 2.0 * pad)


def eval_nurbs_scalar_field(nurbs, u_vec, uv_points):
    """Evaluate the high-fidelity NURBS truth field at arbitrary uv points."""
    u_3d = u_vec.reshape(nurbs.ctrl_u, nurbs.ctrl_v, 1)
    return _eval_nurbs_surface(
        uv_points, u_3d, nurbs.weights,
        nurbs.knotvector_u, nurbs.knotvector_v,
        nurbs.degree_u, nurbs.degree_v
    )[:, 0]


def get_sensors_data(model_p2, truth_nurbs, u_true_vec, n_sensors, noise_std, rng_seed):
    """Build P2 observation matrix and shared physical observations from the truth field."""
    rng = np.random.default_rng(rng_seed)
    uv_sensors = make_sensor_locations(n_sensors, rng_seed)
    H = build_observation_operator(model_p2, uv_sensors)
    u_true_sens = eval_nurbs_scalar_field(truth_nurbs, u_true_vec, uv_sensors)
    u_obs = u_true_sens + rng.normal(0.0, noise_std, size=n_sensors)
    assert H.shape == (n_sensors, model_p2.nodes_2d.shape[0])
    assert u_obs.shape == (n_sensors,)
    return u_obs, H, uv_sensors

# =====================================================================
# Core data assimilation execution logic
# =====================================================================
print("=" * 70)
print("Step 1-4: Assemble high-fidelity NURBS truth system and P2 degraded baseline")
print("=" * 70)


# --- 1. Load NURBS & basic discretization ---
forward_nurbs, elem_spans_forward, _, _ = truth_nurbs(NURBS_NPZ,6,6,50,50)

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


# Truth solution (true reaction coefficient + true source + Robin BC)



F_clean_vol_forward = assemble_nurbs_load(forward_nurbs, elem_spans_forward, f_clean_interp, nq=NQ)

K_forward, M_forward = assemble_nurbs_matrices(forward_nurbs, elem_spans_forward, nq=NQ)
K_robin_forward, F_robin_forward = assemble_nurbs_robin_bc(forward_nurbs, BETA_VAL, U_INF, nq=NQ)

K_true_sys = K_forward + C_TRUTH * M_forward + K_robin_forward
F_true_sys = F_clean_vol_forward + F_robin_forward
# print(np.linalg.cond(K_true_sys),np.linalg.cond(F_true_sys))


u_true_full = spsolve(K_true_sys, F_true_sys)
u_true_int = u_true_full

# 3. Build and load P2 finite element model
p2 = load_p2_from_npz(P2_NPZ)
n_nodes = p2.nodes_2d.shape[0]
print(f"[P2 model info] DOFs = {n_nodes}, mesh = {p2.n_u}x{p2.n_v}")

K_p2, M_p2 = assemble_p2_matrices(p2)
K_robin_p2, F_robin_p2 = assemble_p2_robin_bc(p2, BETA_VAL, U_INF, nq=3)

f_noisy_interp = RegularGridInterpolator((u_vals, v_vals), f_noisy_grid, method='cubic', bounds_error=False, fill_value=None)
F_noisy_vol_p2 = M_p2 @ f_noisy_interp(p2.nodes_2d)

#---------------
F_clean_vol_p2 = M_p2 @ f_clean_interp(p2.nodes_2d)
#---------------

K_model_sys = K_p2 + C_MODEL * M_p2 + K_robin_p2
F_noisy_sys = F_noisy_vol_p2 + F_robin_p2
u_base_full = spsolve(K_model_sys, F_noisy_sys) # Initial P2 FEM solution with missing reaction term

# 4. Evaluate continuous L2 error of Base model using new cross-space integration interface
pre_compute = precompute_l2_error_data(p2,forward_nurbs,u_true_full,NQ)

L2_err_base = compute_l2_error_from_precomputed(pre_compute,u_base_full)
L2_true = np.sqrt(u_true_full @ (M_forward @ u_true_full))
print(f"[Baseline error] Cross-space continuous result: Absolute L2 error between Base field and NURBS truth field = {L2_err_base:.6e}, |u_true| = {L2_true:.6e}")

# 5. Assemble projection transforms needed for assimilation observations and prior params
u_true_3d_block = u_true_full.reshape(forward_nurbs.ctrl_u, forward_nurbs.ctrl_v, 1)
u_true_nodes = _eval_nurbs_surface(p2.nodes_2d, u_true_3d_block, forward_nurbs.weights,
                                   forward_nurbs.knotvector_u, forward_nurbs.knotvector_v,
                                   forward_nurbs.degree_u, forward_nurbs.degree_v)[:, 0]

u_obs, H_full, uv_sensors = get_sensors_data(
    p2, forward_nurbs, u_true_full, N_SENSORS, NOISE_STD, SEED
)
sigma2_y_fixed = NOISE_STD ** 2

# =====================================================================
# Step 5: Eigenvalue low-rank decomposition
# =====================================================================
print("\n" + "=" * 70)
MAX_R = max(R_VALUES)
print(f"Performing generalized eigenvalue decomposition based on P2 baseline stiffness and mass matrices (Max R = {MAX_R})...")
eigvals_all, Vr_all = eigsh(K_model_sys.tocsc(), k=min(MAX_R + 2, n_nodes - 1), M=M_p2.tocsc(), sigma=0.0, which='LM')
order = np.argsort(eigvals_all)
eigvals_all = eigvals_all[order]
Vr_all = Vr_all[:, order]

tol_eig = 1e-12
mask_nz = np.abs(eigvals_all) > tol_eig
eigvals_nz_all = eigvals_all[mask_nz]
Vr_nz_all = Vr_all[:, mask_nz]
print(f"Precomputation done. Found {len(eigvals_nz_all)}  nonzero eigenvalues.")

# =====================================================================
# Step 6 & 7: Statistical assimilation hyperparameter estimation and core solver
# =====================================================================
def weighted_mean(vec, M_mat):
    return float(np.dot(M_mat @ vec, np.ones(len(vec)))) / float(np.sum(M_mat))

def weighted_var(vec, M_mat):
    m = weighted_mean(vec, M_mat)
    return weighted_mean((vec - m)**2, M_mat)

def estimate_hyperparams(K_int, M_int, u_true_int, F_noisy_int, Vr, eigvals_r):
    res = F_noisy_int - K_int @ u_true_int
    sigma2_me = weighted_mean(res**2, M_int)
    var_u = weighted_var(u_true_int, M_int)
    lam1 = eigvals_r[0]
    kappa2 = max(0.5 * lam1, 1e-4)
    tau = var_u * (kappa2 + lam1)
    u_true_anom = u_true_int - weighted_mean(u_true_int, M_int)
    alpha_true = Vr.T @ (M_int @ u_true_anom)
    var_alpha = np.abs(alpha_true)**2 + 1e-8
    Sigma_alpha_inv = np.diag(1.0 / var_alpha)
    return {'sigma2_y': sigma2_y_fixed, 'sigma2_me': sigma2_me, 'kappa': np.sqrt(kappa2), 'tau': tau, 'Sigma_alpha_inv': Sigma_alpha_inv}

def run_assimilation(K_int, M_int, F_noisy_int, Vr, H_int, y_obs, tau, kappa, sigma2_me, sigma2_y, Sigma_alpha_inv):
    n_dof = K_int.shape[0]
    N_obs = H_int.shape[0]
    Q = csc_matrix((1.0 / tau) * (kappa**2 * M_int + K_int))
    R_mat = identity(n_dof, format='csc') / sigma2_me
    Gamma_inv = csc_matrix((1.0 / sigma2_y) * np.eye(N_obs))
    HtGiH = H_int.T @ Gamma_inv @ H_int
    KtRK = K_int.T @ R_mat @ K_int
    Lambda_11 = Q + KtRK + HtGiH
    Lambda_22 = np.asarray(Vr.T @ (HtGiH @ Vr)) + np.asarray(Sigma_alpha_inv)
    Lambda_12 = csc_matrix(HtGiH @ Vr)
    Lambda = bmat([[Lambda_11, Lambda_12], [Lambda_12.T, csc_matrix(Lambda_22)]], format='csc')
    HtGi_y = np.asarray(H_int.T @ (Gamma_inv @ y_obs)).flatten()
    KtR_F = np.asarray(K_int.T @ (R_mat @ F_noisy_int)).flatten()
    b1 = HtGi_y + KtR_F
    b2 = np.asarray(Vr.T @ HtGi_y).flatten()
    sol = spsolve(Lambda, np.concatenate([b1, b2]))
    return sol[:n_dof]

# =====================================================================
# Step 8: Batch run data assimilation and compare truncation effects at different cutoff ranks
# =====================================================================
print("\n" + "=" * 70)
print(f"Running batch R_VALUES optimization assimilation...")
print("=" * 70)

K_csc, M_csc = K_model_sys.tocsc(), M_p2.tocsc()
results_summary = []

for r_val in R_VALUES:
    if r_val > len(eigvals_nz_all): continue
    print(f"\n---> Current truncation mode dimension R_FIXED = {r_val}")
    eigvals_r = eigvals_nz_all[:r_val]
    Vr = Vr_nz_all[:, :r_val]
    
    # A. Statistical analysis estimation
    params_ana = estimate_hyperparams(K_csc, M_csc, u_true_nodes, F_noisy_sys, Vr, eigvals_r)
    u_post_ana = run_assimilation(
        K_csc, M_csc, F_noisy_sys, Vr, H_full, u_obs,
        params_ana['tau'], params_ana['kappa'], params_ana['sigma2_me'],
        params_ana['sigma2_y'], params_ana['Sigma_alpha_inv']
    )
    # Compute analytic assimilation error using new cross-space interface
    L2_err_ana = compute_l2_error_from_precomputed(pre_compute, u_post_ana)
    
    # B. Optuna Bayesian fine optimization
    def objective(trial):
        tau = trial.suggest_float('tau', 1e-1, 1e4, log=True)
        kappa = trial.suggest_float('kappa', 1e-3, 5.0, log=True)
        sigma2_me = trial.suggest_float('sigma2_me', 1e-2, 1e3, log=True)
        alpha_pen = trial.suggest_float('alpha_pen', 1e-6, 1e2, log=True)
        Sigma_alpha_inv = np.eye(r_val) * alpha_pen
        
        u_post = run_assimilation(
            K_csc, M_csc, F_noisy_sys, Vr, H_full, u_obs,
            tau, kappa, sigma2_me, sigma2_y_fixed, Sigma_alpha_inv
        )
        # Optimization objective uses exact cross-space continuous L2 error
        return compute_l2_error_from_precomputed(pre_compute,u_post)

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    t_opt = time.time() - t0
    
    # C. Best solution evaluation
    best = study.best_params
    u_post_opt = run_assimilation(
        K_csc, M_csc, F_noisy_sys, Vr, H_full, u_obs,
        best['tau'], best['kappa'], best['sigma2_me'],
        sigma2_y_fixed, np.eye(r_val) * best['alpha_pen']
    )
    # Compute cross-space L2 error after fine optimization
    L2_err_opt = compute_l2_error_from_precomputed(pre_compute, u_post_opt)
    # errP2 = compute_error_field_as_p2_surface(p2, u_post_opt, forward_nurbs, u_true_full, NQ)
    # visualize_error_field_2d(errP2)
    
    results_summary.append({
        'R': r_val, 'L2_Base': L2_err_base, 'L2_Ana': L2_err_ana, 'L2_Opt': L2_err_opt, 'Time': t_opt
    })
    print(f"  R= {r_val} optimization done | Exact analytical L2: {L2_err_ana:.2e} | Exact Optuna L2: {L2_err_opt:.2e} | Optimization time: {t_opt:.1f}s")

# =====================================================================
# Final strict cross-space data assimilation results table
# =====================================================================
print("\n" + "=" * 95)
print(f"{'R':<6} | {'Continuous L2 (Base)':<22} | {'Continuous L2 (Ana)':<22} | {'Continuous L2 (Opt)':<22} | {'Time(s)':<10}")
print("-" * 95)
for res in results_summary:
    print(f"{res['R']:<6} | {res['L2_Base']:<22.6e} | {res['L2_Ana']:<22.6e} | {res['L2_Opt']:<22.6e} | {res['Time']:<10.1f}")
print("=" * 95)

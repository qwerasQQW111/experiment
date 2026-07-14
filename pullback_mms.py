"""
pullback_mms.py — NURBS Pullback MMS computation library
===========================================
Directly computes the Laplace-Beltrami operator on the parametric domain
 via pullback MMS using rigorous NURBS rational derivative recurrence.

Usage:
    from pullback_mms import compute_mms, compute_mms_from_npz

    # Method 1: existing Surface_NURBS object
    nurbs = load_nurbs_from_npz("wavy_torus_nurbs.npz")
    x, y, z, u, f = compute_mms(nurbs, n_grid=80, k1=2.0, k2=2.0, c=1.0)

    # Method 2: directly from npz file
    x, y, z, u, f = compute_mms_from_npz("wavy_torus_nurbs.npz", n_grid=80)

Theory:
    Manufactured solution: u_tilde(xi,eta) = sin^2(pi*k1*xi) * sin^2(pi*k2*eta)  (Dirichlet compatible)
    Source term:    f = -Δ_g u_tilde + c*u_tilde         (divergence-form Laplace-Beltrami)
"""

import numpy as np
from math import factorial
from pathlib import Path
from surface_model import Surface_NURBS, load_nurbs_from_npz

__all__ = [
    'compute_mms',
    'compute_mms_from_npz',
    'evaluate_nurbs_rational_derivs',
    'basis_functions_derivs_strict',
    'eval_mms_u',
    'simulate_sensors',
    'corrupt_source_term',
]


# ============================================================================
# 1. Rigorous NURBS rational derivative engine (shared single W, not per-order independent normalization)
# ============================================================================

def find_span(u: float, U: np.ndarray, p: int, n: int) -> int:
    """Piegl & Tiller A2.1 — open knot safe version."""
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def basis_functions_derivs_strict(
    u: float, U: np.ndarray, p: int, nderiv: int = 2
) -> tuple:
    """Piegl & Tiller A2.3 strict reference implementation — returns (ders, span).

    ders[k, r] = d^k N_{span-p+r} / du^k,  k = 0..nderiv, r = 0..p.
    """
    n = len(U) - p - 2
    span = find_span(u, U, p, n)

    ndu = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    a = np.zeros((2, p + 1))
    ders = np.zeros((nderiv + 1, p + 1))

    # ---- Cox-de Boor basis function values ----
    ndu[0, 0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    for r in range(p + 1):
        ders[0, r] = ndu[r, p]

    # ---- Derivative recurrence ----
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[s1, 0] = 1.0
        for k in range(1, min(nderiv, p) + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1

    for k in range(1, min(nderiv, p) + 1):
        fac = factorial(p) // factorial(p - k)
        ders[k, :] *= fac

    return ders, span


def evaluate_nurbs_rational_derivs(
    uv_pts: np.ndarray,
    ctrlpts: np.ndarray,
    weights: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    p: int,
    q: int,
) -> dict:
    """Rigorous NURBS rational derivative computation.

    Core principle: all-order derivatives share a single W(u,v) and its derivatives,
    using the Leibniz rule to derive S^(k,l) from weighted control point moments A^(k,l) and weight moments W^(k,l).

    Parameters
    ----------
    uv_pts : (N, 2) parameter coordinate array
    ctrlpts, weights, U, V, p, q : NURBS definition parameters

    Returns
    -------
    dict with keys: 'S', 'Su', 'Sv', 'Suu', 'Suv', 'Svv' — each (N, 3)
    """
    n_pts = uv_pts.shape[0]
    out = {
        'S':   np.zeros((n_pts, 3)),
        'Su':  np.zeros((n_pts, 3)),
        'Sv':  np.zeros((n_pts, 3)),
        'Suu': np.zeros((n_pts, 3)),
        'Suv': np.zeros((n_pts, 3)),
        'Svv': np.zeros((n_pts, 3)),
    }

    for idx in range(n_pts):
        u, v = uv_pts[idx]
        Nu, su = basis_functions_derivs_strict(u, U, p, 2)
        Nv, sv = basis_functions_derivs_strict(v, V, q, 2)

        ui = su - p + np.arange(p + 1)
        vi = sv - q + np.arange(q + 1)
        P_loc = ctrlpts[np.ix_(ui, vi)]       # (p+1, q+1, 3)
        w_loc = weights[np.ix_(ui, vi)]       # (p+1, q+1)

        # ---- Weighted moments A^(du,dv) and weight moments W^(du,dv) ----
        def moment_wP(du, dv):
            return np.einsum('i,j,ijk->k', Nu[du], Nv[dv], w_loc[..., None] * P_loc)

        def moment_w(du, dv):
            return np.einsum('i,j,ij->', Nu[du], Nv[dv], w_loc)

        A00 = moment_wP(0, 0); A10 = moment_wP(1, 0); A01 = moment_wP(0, 1)
        A20 = moment_wP(2, 0); A11 = moment_wP(1, 1); A02 = moment_wP(0, 2)

        W00 = moment_w(0, 0);  W10 = moment_w(1, 0);  W01 = moment_w(0, 1)
        W20 = moment_w(2, 0);  W11 = moment_w(1, 1);  W02 = moment_w(0, 2)

        eps = 1e-30
        W = W00 if abs(W00) > eps else eps

        # ---- Rigorous rational derivative recurrence (Leibniz / quotient rule) ----
        S   = A00 / W
        Su  = (A10 - W10 * S) / W
        Sv  = (A01 - W01 * S) / W
        Suu = (A20 - 2.0 * W10 * Su - W20 * S) / W
        Suv = (A11 - W10 * Sv - W01 * Su - W11 * S) / W
        Svv = (A02 - 2.0 * W01 * Sv - W02 * S) / W

        out['S'][idx]   = S
        out['Su'][idx]  = Su
        out['Sv'][idx]  = Sv
        out['Suu'][idx] = Suu
        out['Suv'][idx] = Suv
        out['Svv'][idx] = Svv

    return out


# ============================================================================
# 2. Pullback MMS core computation
# ============================================================================

def compute_mms(
    nurbs: Surface_NURBS,
    n_grid: int = 80,
    k1: float = 2.0,
    k2: float = 2.0,
    c: float = 1.0,
) -> tuple:
    """Non-periodic surface Pullback MMS.

    Manufactured solution: u_tilde(xi,eta) = sin^2(pi*k1*xi) * sin^2(pi*k2*eta),
    at open knot boundaries u_tilde=0, grad(u_tilde)=0 (Dirichlet compatible). Computes Laplace-Beltrami via pullback metric on parametric domain.

    Parameters
    ----------
    nurbs : Surface_NURBS
        NURBS surface object.
    n_grid : int
        Grid points per direction (after boundary pad).
    k1, k2 : float
        Frequency factors for manufactured solution in xi and eta directions.
    c : float
        Zeroth-order term coefficient: f = -Δ_g u_tilde + c*u_tilde.

    Returns
    -------
    x, y, z : (n_grid, n_grid) ndarray
        Physical space coordinates (for visualization only).
    u : (n_grid, n_grid) ndarray
        Manufactured solution u_tilde(xi,eta).
    f : (n_grid, n_grid) ndarray
        Source term f = -Δ_g u_tilde + c*u_tilde.
    """
    pad = 1e-6
    xi = np.linspace(pad, 1.0 - pad, n_grid)
    eta = np.linspace(pad, 1.0 - pad, n_grid)
    Xi, Eta = np.meshgrid(xi, eta, indexing='ij')
    uv_pts = np.stack([Xi.ravel(), Eta.ravel()], axis=-1)

    # 1. Rigorous rational geometry derivatives
    derivs = evaluate_nurbs_rational_derivs(
        uv_pts, nurbs.control_points, nurbs.weights,
        nurbs.knotvector_u, nurbs.knotvector_v,
        nurbs.degree_u, nurbs.degree_v,
    )
    Su, Sv = derivs['Su'], derivs['Sv']
    Suu, Suv, Svv = derivs['Suu'], derivs['Suv'], derivs['Svv']

    # 2. Manufactured solution and its derivatives (Dirichlet compatible)
    K1, K2 = np.pi * k1, np.pi * k2
    s1 = np.sin(K1 * Xi.ravel()); c1 = np.cos(K1 * Xi.ravel())
    s2 = np.sin(K2 * Eta.ravel()); c2 = np.cos(K2 * Eta.ravel())

    u_val    = (s1 ** 2) * (s2 ** 2)
    u_xi     = 2.0 * K1 * s1 * c1 * (s2 ** 2)
    u_eta    = 2.0 * K2 * (s1 ** 2) * s2 * c2
    u_xixi   = 2.0 * K1 ** 2 * (c1 ** 2 - s1 ** 2) * (s2 ** 2)
    u_etaeta = 2.0 * K2 ** 2 * (s1 ** 2) * (c2 ** 2 - s2 ** 2)
    u_xieta  = 4.0 * K1 * K2 * s1 * c1 * s2 * c2

    # 3. Laplace-Beltrami (Divergence Form) — pullback to parametric domain
    guu = np.einsum('ni,ni->n', Su, Su)
    guv = np.einsum('ni,ni->n', Su, Sv)
    gvv = np.einsum('ni,ni->n', Sv, Sv)

    det_g = guu * gvv - guv ** 2
    sqrt_g = np.sqrt(np.maximum(det_g, 1e-30))

    gUU = gvv / det_g
    gUV = -guv / det_g
    gVV = guu / det_g

    # Inverse metric derivatives (chain rule)
    guu_u = 2.0 * np.einsum('ni,ni->n', Suu, Su)
    guu_v = 2.0 * np.einsum('ni,ni->n', Suv, Su)
    guv_u = np.einsum('ni,ni->n', Suu, Sv) + np.einsum('ni,ni->n', Su, Suv)
    guv_v = np.einsum('ni,ni->n', Suv, Sv) + np.einsum('ni,ni->n', Su, Svv)
    gvv_u = 2.0 * np.einsum('ni,ni->n', Suv, Sv)
    gvv_v = 2.0 * np.einsum('ni,ni->n', Svv, Sv)

    ddet_u = guu_u * gvv + guu * gvv_u - 2.0 * guv * guv_u
    ddet_v = guu_v * gvv + guu * gvv_v - 2.0 * guv * guv_v

    det2 = det_g ** 2 + 1e-60
    dgUU_u = (gvv_u * det_g - gvv * ddet_u) / det2
    dgUU_v = (gvv_v * det_g - gvv * ddet_v) / det2
    dgUV_u = (-guv_u * det_g + guv * ddet_u) / det2
    dgUV_v = (-guv_v * det_g + guv * ddet_v) / det2
    dgVV_u = (guu_u * det_g - guu * ddet_u) / det2
    dgVV_v = (guu_v * det_g - guu * ddet_v) / det2

    dsqrtg_u = ddet_u / (2.0 * sqrt_g + 1e-30)
    dsqrtg_v = ddet_v / (2.0 * sqrt_g + 1e-30)

    F_xi  = sqrt_g * (gUU * u_xi + gUV * u_eta)
    F_eta = sqrt_g * (gUV * u_xi + gVV * u_eta)

    dF_xi = (dsqrtg_u * (gUU * u_xi + gUV * u_eta) +
             sqrt_g * (dgUU_u * u_xi + gUU * u_xixi + dgUV_u * u_eta + gUV * u_xieta))
    dF_eta = (dsqrtg_v * (gUV * u_xi + gVV * u_eta) +
              sqrt_g * (dgUV_v * u_xi + gUV * u_xieta + dgVV_v * u_eta + gVV * u_etaeta))

    lap_u = (dF_xi + dF_eta) / (sqrt_g + 1e-30)
    f_val = -lap_u + c * u_val

    # 4. Reshape to (n_grid, n_grid)
    shape = (n_grid, n_grid)
    x = derivs['S'][:, 0].reshape(shape)
    y = derivs['S'][:, 1].reshape(shape)
    z = derivs['S'][:, 2].reshape(shape)
    u = u_val.reshape(shape)
    f = f_val.reshape(shape)

    return x, y, z, u, f


def compute_mms_from_npz(
    npz_path: str | Path,
    n_grid: int = 80,
    k1: float = 2.0,
    k2: float = 2.0,
    c: float = 1.0,
) -> tuple:
    """Load NURBS surface from .npz file and compute Pullback MMS.

    Parameters
    ----------
    npz_path : str or Path
        NURBS .npz file path (generated by truth_to_nurbs).
    n_grid : int
        Grid points per direction.
    k1, k2 : float
        Manufactured solution frequency factors.
    c : float
        Zeroth-order term coefficient.

    Returns
    -------
    x, y, z, u, f : same as compute_mms.
    """
    nurbs = load_nurbs_from_npz(npz_path)
    return compute_mms(nurbs, n_grid=n_grid, k1=k1, k2=k2, c=c)


# ============================================================================
# 3. MMS manufactured solution point evaluation & sensor simulation
# ============================================================================

def eval_mms_u(
    uv_pts: np.ndarray,
    k1: float = 2.0,
    k2: float = 2.0,
) -> np.ndarray:
    """Evaluate manufactured solution u_tilde(xi,eta) at arbitrary parametric coordinates.

    u_tilde(xi,eta) = sin^2(pi*k1*xi) * sin^2(pi*k2*eta)

    Parameters
    ----------
    uv_pts : (N, 2) ndarray
        Parameter coordinates (xi_i, eta_i) in (0,1)^2.
    k1, k2 : float
        Manufactured solution frequency factors.

    Returns
    -------
    u : (N,) ndarray
        Manufactured solution values.
    """
    K1, K2 = np.pi * k1, np.pi * k2
    xi = uv_pts[:, 0]
    eta = uv_pts[:, 1]
    s1 = np.sin(K1 * xi)
    s2 = np.sin(K2 * eta)
    return (s1 ** 2) * (s2 ** 2)


def simulate_sensors(
    nurbs: Surface_NURBS,
    n_sensors: int = 9,
    noise_std: float = 0.0,
    k1: float = 2.0,
    k2: float = 2.0,
    rng: np.random.Generator | int | None = 42,
) -> tuple:
    """Place n_sensors sensors on NURBS surface, return physical coordinates and noisy u observations.

    Sensors distributed in a roughly 3x3 grid inside the parametric domain (avoiding open knot boundary pad),
    if n_sensors != 9 use quasi-random Halton sampling.

    Parameters
    ----------
    nurbs : Surface_NURBS
        NURBS surface object.
    n_sensors : int
        Number of sensors (default 9, arranged in 3x3 grid).
    noise_std : float
        Additive Gaussian noise std dev sigma. sigma=0 means noise-free truth.
    k1, k2 : float
        Manufactured solution frequency factors.
    rng : int, np.random.Generator or None
        Random seed / Generator. None means no seeding.

    Returns
    -------
    phys_coords : (n_sensors, 3) ndarray
        Sensor physical coordinates S(xi_i, eta_i).
    u_obs : (n_sensors,) ndarray
        Noisy observations: u_obs = u_tilde(xi_i, eta_i) + eps_i,  eps_i ~ N(0, sigma^2).
    u_true : (n_sensors,) ndarray
        Noise-free truth u_tilde(xi_i, eta_i).
    uv_sensors : (n_sensors, 2) ndarray
        Sensor parametric coordinates (xi_i, eta_i).
    """
    # -- Build RNG --
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    # -- Determine sensor parametric coordinates --
    pad = 1e-6
    if n_sensors == 9:
        # 3x3 uniform grid, avoiding boundary
        lin = np.linspace(pad, 1.0 - pad, 5)[1:-1]  # 3 interior points
        Xi, Eta = np.meshgrid(lin, lin, indexing='ij')
        uv_sensors = np.stack([Xi.ravel(), Eta.ravel()], axis=-1)
    else:
        # Quasi-random Halton sampling for uniform coverage
        from scipy.stats import qmc
        sampler = qmc.Halton(d=2, seed=rng.integers(0, 2**31))
        uv_sensors = sampler.random(n_sensors)
        uv_sensors = pad + uv_sensors * (1.0 - 2.0 * pad)

    # -- Evaluate NURBS to get physical coordinates --
    derivs = evaluate_nurbs_rational_derivs(
        uv_sensors, nurbs.control_points, nurbs.weights,
        nurbs.knotvector_u, nurbs.knotvector_v,
        nurbs.degree_u, nurbs.degree_v,
    )
    phys_coords = derivs['S']   # (n_sensors, 3)

    # -- Evaluate truth --
    u_true = eval_mms_u(uv_sensors, k1=k1, k2=k2)  # (n_sensors,)

    # -- Add noise --
    if noise_std > 0:
        u_obs = u_true + rng.normal(0.0, noise_std, size=n_sensors)
    else:
        u_obs = u_true.copy()

    return phys_coords, u_obs, u_true, uv_sensors


# ============================================================================
# 4. Source term f noise corruption (white noise + low-frequency noise)
# ============================================================================

def corrupt_source_term(
    f: np.ndarray,
    n_grid: int,
    white_std: float = 0.0,
    low_freq_amplitude: float = 0.0,
    low_freq_modes: int = 5,
    rng: np.random.Generator | int | None = 42,
) -> tuple:
    """Add white and low-frequency noise to manufactured source term f, simulating real observation degradation.

    Noise model:
        f_obs(ξ,η) = f(ξ,η) + ε_w(ξ,η) + ε_lf(ξ,η)

    - eps_w: white noise — pointwise independent Gaussian, N(0, white_std^2)
            simulates sensor measurement noise / high-freq numerical error.

    - eps_lf: low-freq noise — random superposition of a few low-order Fourier modes,
            simulates smooth, spatially correlated structural error
            caused by model misspecification.

    Parameters
    ----------
    f : (n_grid, n_grid) ndarray
        Original (noise-free) source term.
    n_grid : int
        Parametric domain grid resolution (must match f shape).
    white_std : float
        White noise std dev sigma_w. sigma_w=0 skips white noise.
    low_freq_amplitude : float
        Low-freq noise amplitude A. Each modal coefficient a_i ~ N(0, A^2).
        A=0 skips low-freq noise.
    low_freq_modes : int
        Number of low-freq Fourier modes. Each mode is:
            cos(pi*m_i*xi + phi_xi_i) * cos(pi*n_i*eta + phi_eta_i)
        where m_i, n_i in {0, 1, 2} ensures low wavenumber.
    rng : int, np.random.Generator or None
        Random seed.

    Returns
    -------
    f_noisy : (n_grid, n_grid) ndarray
        Corrupted source term f + eps_w + eps_lf.
    f_white : (n_grid, n_grid) ndarray
        Pure white noise component eps_w (diagnostic).
    f_low_freq : (n_grid, n_grid) ndarray
        Pure low-freq noise component eps_lf (diagnostic).
    """
    if f.shape != (n_grid, n_grid):
        raise ValueError(f"f shape {f.shape} != (n_grid, n_grid) = ({n_grid},{n_grid})")

    # -- Build RNG --
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    # Parametric domain grid (consistent with pad in compute_mms)
    pad = 1e-6
    xi = np.linspace(pad, 1.0 - pad, n_grid)
    eta = np.linspace(pad, 1.0 - pad, n_grid)
    Xi, Eta = np.meshgrid(xi, eta, indexing='ij')

    # ---------- White noise ----------
    if white_std > 0:
        f_white = rng.normal(0.0, white_std, size=(n_grid, n_grid))
    else:
        f_white = np.zeros_like(f)

    # ---------- Low-frequency noise ----------
    if low_freq_amplitude > 0:
        f_low_freq = np.zeros_like(f)
        # Low wavenumber candidates: Cartesian product of {0, 1, 2}
        low_modes = [(m, n) for m in range(3) for n in range(3) if not (m == 0 and n == 0)]
        # Sample low_freq_modes without replacement
        chosen = low_freq_modes
        if chosen > len(low_modes):
            chosen = len(low_modes)
        indices = rng.choice(len(low_modes), size=min(chosen, len(low_modes)), replace=False)
        for idx in indices:
            m, n = low_modes[idx]
            a = rng.normal(0.0, low_freq_amplitude)
            phi_xi = rng.uniform(0, 2.0 * np.pi)
            phi_eta = rng.uniform(0, 2.0 * np.pi)
            f_low_freq += a * np.cos(np.pi * m * Xi + phi_xi) * np.cos(np.pi * n * Eta + phi_eta)
    else:
        f_low_freq = np.zeros_like(f)

    f_noisy = f + f_white + f_low_freq

    return f_noisy, f_white, f_low_freq

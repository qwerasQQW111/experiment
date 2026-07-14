"""
surface_model.py — Unified geometry modeling library for wavy torus
====================================================================
Provides four data structures and seven core operations covering the full
geometry pipeline from analytic truth model to NURBS representation to
P1 / P2 finite element discretization.

Data structures:
  Surface_Truth  — Analytic T^2 torus with multi-frequency sine wave perturbation
  Surface_NURBS  — NURBS (B-spline) surface representation
  Surface_P1     — P1 linear Lagrange triangular element discretization
  Surface_P2     — P2 quadratic Lagrange triangular element discretization (6 nodes per triangle)

Core functions:
  init_truth_model()      — Initialize truth model (Surface_Truth)
  truth_to_nurbs()        — Truth -> NURBS fit -> save .npz
  load_nurbs_from_npz()   — Restore Surface_NURBS from .npz
  nurbs_to_p1()           — NURBS -> P1 discretization -> save .npz
  nurbs_to_p2()           — NURBS -> P2 discretization -> save .npz
  load_p1_from_npz()      — Restore Surface_P1 from .npz
  load_p2_from_npz()      — Restore Surface_P2 from .npz

Dependencies: numpy, geomdl (NURBS-Python)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


# ============================================================================
# 1. Data structure definitions
# ============================================================================

@dataclass
class Surface_Truth:
    """Analytic T^2 torus + multi-frequency sine wave perturbation.

    Parameterization: (u,v) in [0, 2pi)^2
    Mathematical definition:
        x(u,v) = (R + r_eff(u,v)*cos(v)) * cos(u)
        y(u,v) = (R + r_eff(u,v)*cos(v)) * sin(u)
        z(u,v) = r_eff(u,v) * sin(v)
        r_eff(u,v) = r + sum_i A_i*sin(omega_i*u+phi_ui)*sin(nu_i*v+phi_vi)
    """
    R: float                                    # major radius
    r: float                                    # base minor radius
    wave_components: list                       # [(A, omega_u, nu_v, phi_u, phi_v), ...]
    n_u: int                                    # u-direction sampling resolution
    n_v: int                                    # v-direction sampling resolution
    x: np.ndarray = field(default=None, repr=False)   # (n_u, n_v) 3D coordinates
    y: np.ndarray = field(default=None, repr=False)
    z: np.ndarray = field(default=None, repr=False)
    u: np.ndarray = field(default=None, repr=False)   # (n_u,) parameter coordinates
    v: np.ndarray = field(default=None, repr=False)   # (n_v,)
    u_grid: np.ndarray = field(default=None, repr=False)  # (n_u, n_v) parameter grid
    v_grid: np.ndarray = field(default=None, repr=False)
    delta: np.ndarray = field(default=None, repr=False)  # (n_u, n_v) perturbation field


@dataclass
class Surface_NURBS:
    """NURBS (B-spline) surface representation.

    Parameterization: (u,v) in [0, 1]^2 (geomdl normalized knot vectors)

    S(u,v) = sum_i sum_j N_i^p(u) N_j^q(v) w_ij P_ij / sum_i sum_j N_i^p(u) N_j^q(v) w_ij
    """
    control_points: np.ndarray                   # (ctrl_u, ctrl_v, 3)
    weights: np.ndarray                          # (ctrl_u, ctrl_v)
    knotvector_u: np.ndarray                     # (n_knots_u,)
    knotvector_v: np.ndarray                     # (n_knots_v,)
    degree_u: int
    degree_v: int
    ctrl_u: int
    ctrl_v: int
    torus_R: float = 3.0
    torus_r: float = 1.0

    def eval(self, uv: np.ndarray) -> np.ndarray:
        """Evaluate NURBS surface at (N, 2) parameter points, returning (N, 3) 3D coordinates."""
        return _eval_nurbs_surface(
            uv, self.control_points, self.weights,
            self.knotvector_u, self.knotvector_v,
            self.degree_u, self.degree_v,
        )


@dataclass
class Surface_P1:
    """P1 linear Lagrange triangular element discretization.

    Structured triangular mesh on parametric domain [0,1]^2, each rectangle
    split diagonally into 2 triangles, 3 vertex nodes per triangle,
    piecewise linear interpolation.
    """
    nodes_2d: np.ndarray                         # (n_nodes, 2) — parametric coordinates
    nodes_3d: np.ndarray                         # (n_nodes, 3) — 3D coordinates
    triangles: np.ndarray                        # (n_elems, 3) — vertex indices
    n_u: int                                     # u-direction divisions
    n_v: int                                     # v-direction divisions
    h: float                                     # mesh size = 1/n_div
    degree: int = field(default=1, init=False)


@dataclass
class Surface_P2:
    """P2 quadratic Lagrange triangular element discretization.

    Structured triangular mesh on parametric domain [0,1]^2, 6 nodes per triangle
    (3 vertices + 3 edge midpoints), piecewise quadratic interpolation.
    Node numbering order:
      [0, n_vert)                   — vertices
      [n_vert, n_vert+n_h)          — horizontal edge midpoints
      [n_vert+n_h, n_vert+n_h+n_ve) — vertical edge midpoints
      [n_vert+n_h+n_ve, total)      — diagonal midpoints
    """
    nodes_2d: np.ndarray                         # (n_nodes, 2)
    nodes_3d: np.ndarray                         # (n_nodes, 3)
    triangles: np.ndarray                        # (n_elems, 6) — [v1,v2,v3,mid12,mid23,mid13]
    n_u: int
    n_v: int
    h: float
    degree: int = field(default=2, init=False)


# ============================================================================
# 2. Internal: B-spline surface evaluation (Cox-de Boor algorithm)
# ============================================================================

def _find_spans(n: int, p: int, u_arr: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Vectorized knot span index lookup (Algorithm A2.1, NURBS Book).

    n = n_ctrl - 1, p = degree.
    For clamped knot vectors, returned span in [p, n].
    """
    spans = np.searchsorted(U, u_arr, side='right') - 1
    return np.clip(spans, p, n)


def _basis_funs(spans: np.ndarray, u_arr: np.ndarray, p: int,
                U: np.ndarray) -> np.ndarray:
    """Vectorized B-spline basis function evaluation (Algorithm A2.2, NURBS Book).

    Returns (N, p+1) — each row contains p+1 nonzero basis functions N_{i-p},...,N_i.
    """
    m = len(u_arr)
    N = np.zeros((m, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    for k in range(m):
        span = spans[k]
        u = u_arr[k]
        N[k, 0] = 1.0
        for j in range(1, p + 1):
            left[j] = u - U[span + 1 - j]
            right[j] = U[span + j] - u
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = N[k, r] / denom if abs(denom) > 1e-15 else 0.0
                N[k, r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[k, j] = saved
    return N


def _eval_nurbs_surface(points_uv: np.ndarray,
                        ctrlpts: np.ndarray,
                        weights: np.ndarray,
                        U: np.ndarray, V: np.ndarray,
                        p: int, q: int) -> np.ndarray:
    """Evaluate rational B-spline surface at arbitrary (u, v) point sets.

    Parameters
    ----------
    points_uv : (N, 2) — (u, v) parametric coordinates
    ctrlpts   : (n_ctrl_u, n_ctrl_v, 3) — control points
    weights   : (n_ctrl_u, n_ctrl_v) — weights (all 1 for B-spline)
    U, V      : u/v knot vectors
    p, q      : u/v degrees

    Returns
    -------
    result : (N, 3) — 3D points on the surface
    """
    u_arr = points_uv[:, 0]
    v_arr = points_uv[:, 1]
    n_ctrl_u, n_ctrl_v = ctrlpts.shape[:2]

    spans_u = _find_spans(n_ctrl_u - 1, p, u_arr, U)
    spans_v = _find_spans(n_ctrl_v - 1, q, v_arr, V)

    Nu = _basis_funs(spans_u, u_arr, p, U)  # (N, p+1)
    Nv = _basis_funs(spans_v, v_arr, q, V)  # (N, q+1)

    # Collect local control points / weights for each point
    u_idx = spans_u[:, None] - p + np.arange(p + 1)[None, :]   # (N, p+1)
    v_idx = spans_v[:, None] - q + np.arange(q + 1)[None, :]   # (N, q+1)

    local_pts = ctrlpts[u_idx[:, :, None], v_idx[:, None, :]]  # (N, p+1, q+1, 3)
    local_w = weights[u_idx[:, :, None], v_idx[:, None, :]]    # (N, p+1, q+1)

    # Tensor product basis
    NuNv = np.einsum('na,nb->nab', Nu, Nv)                    # (N, p+1, q+1)

    # Rational formula: S = sum w N P / sum w N
    denom = np.sum(NuNv * local_w, axis=(1, 2))               # (N,)
    numer = np.einsum('nab,nabc->nc', NuNv * local_w, local_pts)  # (N, 3)

    result = numer / denom[:, None]
    return result


# ============================================================================
# 3. Internal: Mesh construction
# ============================================================================

def _build_p1_mesh(n_u: int, n_v: int):
    """Build P1 structured triangular mesh on [0,1]^2.

    Returns
    -------
    nodes_2d  : ((n_u+1)*(n_v+1), 2) — vertex parameter coordinates
    triangles : (2*n_u*n_v, 3) — triangle vertex indices
    """
    u_nodes = np.linspace(0, 1, n_u + 1)
    v_nodes = np.linspace(0, 1, n_v + 1)
    u_grid, v_grid = np.meshgrid(u_nodes, v_nodes, indexing='ij')
    nodes_2d = np.stack([u_grid.ravel(), v_grid.ravel()], axis=-1)

    # Vectorized triangle connectivity construction
    i_arr, j_arr = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing='ij')

    v00 = j_arr       + i_arr       * (n_v + 1)
    v10 = j_arr       + (i_arr + 1) * (n_v + 1)
    v11 = (j_arr + 1) + (i_arr + 1) * (n_v + 1)
    v01 = (j_arr + 1) + i_arr       * (n_v + 1)

    tri1 = np.stack([v00, v10, v11], axis=-1)  # lower triangle
    tri2 = np.stack([v00, v11, v01], axis=-1)  # upper triangle

    triangles = np.stack([tri1, tri2], axis=1).reshape(-1, 3)
    return nodes_2d, triangles


def _build_p2_mesh(n_u: int, n_v: int):
    """Build P2 structured triangular mesh on [0,1]^2.

    P2 nodes = vertices + horizontal edge midpoints + vertical edge midpoints + diagonal midpoints.

    6-node triangle index order: [v1, v2, v3, mid12, mid23, mid13]

    Returns
    -------
    nodes_2d  : (N_total, 2)
    triangles : (2*n_u*n_v, 6)
    """
    u_nodes = np.linspace(0, 1, n_u + 1)
    v_nodes = np.linspace(0, 1, n_v + 1)
    u_h = (u_nodes[:-1] + u_nodes[1:]) / 2   # horizontal edge midpoint u coords
    v_h = (v_nodes[:-1] + v_nodes[1:]) / 2   # vertical edge midpoint v coords

    # Vertices
    ug, vg = np.meshgrid(u_nodes, v_nodes, indexing='ij')
    vert_2d = np.stack([ug.ravel(), vg.ravel()], axis=-1)          # ((n_u+1)*(n_v+1), 2)

    # Horizontal edge midpoints: (u_h[i], v_nodes[j])
    ug, vg = np.meshgrid(u_h, v_nodes, indexing='ij')
    horiz_2d = np.stack([ug.ravel(), vg.ravel()], axis=-1)         # (n_u*(n_v+1), 2)

    # Vertical edge midpoints: (u_nodes[i], v_h[j])
    ug, vg = np.meshgrid(u_nodes, v_h, indexing='ij')
    vert_e_2d = np.stack([ug.ravel(), vg.ravel()], axis=-1)        # ((n_u+1)*n_v, 2)

    # Diagonal midpoints: (u_h[i], v_h[j])
    ug, vg = np.meshgrid(u_h, v_h, indexing='ij')
    diag_2d = np.stack([ug.ravel(), vg.ravel()], axis=-1)          # (n_u*n_v, 2)

    nodes_2d = np.vstack([vert_2d, horiz_2d, vert_e_2d, diag_2d])

    # Triangle connectivity
    n_vert   = (n_u + 1) * (n_v + 1)
    n_h      = n_u * (n_v + 1)
    n_v_edge = (n_u + 1) * n_v

    i_arr, j_arr = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing='ij')

    # Vertex indices
    v00 = j_arr       + i_arr       * (n_v + 1)
    v10 = j_arr       + (i_arr + 1) * (n_v + 1)
    v11 = (j_arr + 1) + (i_arr + 1) * (n_v + 1)
    v01 = (j_arr + 1) + i_arr       * (n_v + 1)

    # Midpoint indices
    h00   = n_vert + j_arr       + i_arr       * (n_v + 1)           # horiz(i, j)
    h01   = n_vert + (j_arr + 1) + i_arr       * (n_v + 1)           # horiz(i, j+1)
    v10_e = n_vert + n_h + j_arr + (i_arr + 1) * n_v                 # vert(i+1, j)
    v00_e = n_vert + n_h + j_arr + i_arr       * n_v                 # vert(i, j)
    d00   = n_vert + n_h + n_v_edge + j_arr + i_arr * n_v            # diag(i, j)

    # Triangle 1: [v00, v10, v11, h00, v10_e, d00]
    tri1 = np.stack([v00, v10, v11, h00, v10_e, d00], axis=-1)
    # Triangle 2: [v00, v11, v01, d00, h01, v00_e]
    tri2 = np.stack([v00, v11, v01, d00, h01, v00_e], axis=-1)

    triangles = np.stack([tri1, tri2], axis=1).reshape(-1, 6)
    return nodes_2d, triangles


# ============================================================================
# 4. Public API — Initialize truth model
# ============================================================================

def init_truth_model(R: float = 3.0,
                     r: float = 1.0,
                     wave_components: list | None = None,
                     n_u: int = 300,
                     n_v: int = 300) -> Surface_Truth:
    """Initialize the T^2 torus + multi-frequency sine wave perturbation truth model.

    Parameters
    ----------
    R : major radius (torus center to tube center), default 3.0
    r : base minor radius (tube cross-section radius), default 1.0
    wave_components : sine wave component list [(A, omega_u, nu_v, phi_u, phi_v), ...]
                      defaults to empty (smooth torus)
    n_u, n_v : sampling resolution, default 300

    Returns
    -------
    Surface_Truth — data structure with full perturbation field and 3D coordinates
    """
    if wave_components is None:
        wave_components = [
            # (0.15,  3,  2,  0.0,      0.0     ),
            # (0.20,  5,  4,  np.pi/4,  np.pi/3 ),
            # (0.12,  8,  7,  np.pi/2,  np.pi/6 ),
            # (0.07, 14, 11,  np.pi/3,  np.pi/4 ),
            # (0.04, 22, 18,  np.pi/5,  np.pi/7 ),
        ]

    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    u_grid, v_grid = np.meshgrid(u, v)

    # Compute perturbation
    delta = np.zeros_like(u_grid)
    for amp, fu, fv, pu, pv in wave_components:
        delta += amp * np.sin(fu * u_grid + pu) * np.sin(fv * v_grid + pv)

    r_eff = r + delta

    x = (R + r_eff * np.cos(v_grid)) * np.cos(u_grid)
    y = (R + r_eff * np.cos(v_grid)) * np.sin(u_grid)
    z = r_eff * np.sin(v_grid)

    return Surface_Truth(
        R=R, r=r, wave_components=wave_components,
        n_u=n_u, n_v=n_v,
        x=x, y=y, z=z, u=u, v=v,
        u_grid=u_grid, v_grid=v_grid, delta=delta,
    )


# ============================================================================
# 5. Public API — Truth -> NURBS -> save npz
# ============================================================================

def truth_to_nurbs(truth: Surface_Truth,
                   output_path: str | Path,
                   degree_u: int = 3,
                   degree_v: int = 3,
                   ctrl_u: int = 40,
                   ctrl_v: int = 40,
                   samp_u: int = 120,
                   samp_v: int = 120) -> Surface_NURBS:
    """Convert Surface_Truth to NURBS surface representation and save as .npz.

    Uses geomdl global least-squares surface approximation (Algorithm A9.7).

    Parameters
    ----------
    truth       : Surface_Truth — truth model
    output_path : output .npz path
    degree_u, degree_v : NURBS degree, default 3
    ctrl_u, ctrl_v     : u/v direction control point count, default 40
    samp_u, samp_v     : fitting sample point count, default 120

    Returns
    -------
    Surface_NURBS
    """
    from geomdl import fitting

    output_path = Path(output_path)

    # Generate [0,2pi]^2 sample points (geomdl accepts 3D points in any parameter domain)
    u = np.linspace(0, 2 * np.pi, samp_u)
    v = np.linspace(0, 2 * np.pi, samp_v)
    u_grid, v_grid = np.meshgrid(u, v)

    delta = np.zeros_like(u_grid)
    for amp, fu, fv, pu, pv in truth.wave_components:
        delta += amp * np.sin(fu * u_grid + pu) * np.sin(fv * v_grid + pv)

    r_eff = truth.r + delta
    x = (truth.R + r_eff * np.cos(v_grid)) * np.cos(u_grid)
    y = (truth.R + r_eff * np.cos(v_grid)) * np.sin(u_grid)
    z = r_eff * np.sin(v_grid)

    points_flat = np.stack([x, y, z], axis=-1).reshape(-1, 3).tolist()

    # geomdl global least-squares surface approximation
    surf = fitting.approximate_surface(
        points_flat,
        samp_u, samp_v,
        degree_u, degree_v,
        ctrlpts_size_u=ctrl_u,
        ctrlpts_size_v=ctrl_v,
    )

    ctrlpts_2d = surf.ctrlpts2d
    ctrlpts_np = np.array(ctrlpts_2d)           # (ctrl_u, ctrl_v, 3)
    weights_np = np.ones((ctrl_u, ctrl_v))       # B-spline -> all weights = 1
    knot_u = np.array(surf.knotvector_u)
    knot_v = np.array(surf.knotvector_v)

    # Save
    np.savez_compressed(
        output_path,
        control_points=ctrlpts_np,
        weights=weights_np,
        knotvector_u=knot_u,
        knotvector_v=knot_v,
        degree_u=np.array([degree_u]),
        degree_v=np.array([degree_v]),
        torus_R=np.array([truth.R]),
        torus_r=np.array([truth.r]),
        ctrl_u=np.array([ctrl_u]),
        ctrl_v=np.array([ctrl_v]),
    )

    return Surface_NURBS(
        control_points=ctrlpts_np,
        weights=weights_np,
        knotvector_u=knot_u,
        knotvector_v=knot_v,
        degree_u=degree_u,
        degree_v=degree_v,
        ctrl_u=ctrl_u,
        ctrl_v=ctrl_v,
        torus_R=truth.R,
        torus_r=truth.r,
    )

# ============================================================================
# 6. Public API — Load NURBS from npz
# ============================================================================

def load_nurbs_from_npz(npz_path: str | Path) -> Surface_NURBS:
    """Restore NURBS surface representation from .npz file.

    Parameters
    ----------
    npz_path : .npz file path

    Returns
    -------
    Surface_NURBS
    """
    data = np.load(npz_path, allow_pickle=False)

    ctrlpts = data['control_points']
    weights = data['weights']
    knot_u  = data['knotvector_u']
    knot_v  = data['knotvector_v']
    deg_u   = int(data['degree_u'][0])
    deg_v   = int(data['degree_v'][0])
    R       = float(data.get('torus_R', np.array([3.0]))[0])
    r       = float(data.get('torus_r', np.array([1.0]))[0])
    cu      = int(data.get('ctrl_u', np.array([ctrlpts.shape[0]]))[0])
    cv      = int(data.get('ctrl_v', np.array([ctrlpts.shape[1]]))[0])

    return Surface_NURBS(
        control_points=ctrlpts,
        weights=weights,
        knotvector_u=knot_u,
        knotvector_v=knot_v,
        degree_u=deg_u,
        degree_v=deg_v,
        ctrl_u=cu,
        ctrl_v=cv,
        torus_R=R,
        torus_r=r,
    )


# ============================================================================
# 7. Public API — NURBS -> P1 discretization -> save npz
# ============================================================================

def nurbs_to_p1(nurbs: Surface_NURBS,
                output_path: str | Path,
                n_div: int = 48) -> Surface_P1:
    """Convert NURBS surface to P1 linear FEM discretization and save as .npz.

    Builds an n_div × n_div structured triangular mesh on parametric domain [0,1]^2,
    evaluates the NURBS surface exactly at each mesh node.

    Parameters
    ----------
    nurbs       : Surface_NURBS
    output_path : output .npz path
    n_div       : mesh divisions per direction, default 48

    Returns
    -------
    Surface_P1
    """
    n_u = n_v = n_div
    h = 1.0 / n_div

    # Build mesh
    nodes_2d, triangles = _build_p1_mesh(n_u, n_v)
    n_nodes = nodes_2d.shape[0]
    n_elem = triangles.shape[0]

    # Evaluate NURBS at mesh nodes (nodal interpolation)
    nodes_3d = nurbs.eval(nodes_2d)

    # Save
    np.savez_compressed(
        Path(output_path),
        method=np.array(['P1']),
        degree=np.array([1]),
        n_u=np.array([n_u]),
        n_v=np.array([n_v]),
        mesh_h=np.array([h]),
        n_nodes=np.array([n_nodes]),
        n_elements=np.array([n_elem]),
        nodes_2d=nodes_2d,
        nodes_3d=nodes_3d,
        triangles=triangles,
    )

    return Surface_P1(
        nodes_2d=nodes_2d,
        nodes_3d=nodes_3d,
        triangles=triangles,
        n_u=n_u,
        n_v=n_v,
        h=h,
    )


# ============================================================================
# 8. Public API — NURBS -> P2 discretization -> save npz
# ============================================================================

def nurbs_to_p2(nurbs: Surface_NURBS,
                output_path: str | Path,
                n_div: int = 48) -> Surface_P2:
    """Convert NURBS surface to P2 quadratic FEM discretization and save as .npz.

    Builds an n_div × n_div structured P2 triangular mesh on parametric domain
    [0,1]^2 (6 nodes per triangle: 3 vertices + 3 edge midpoints), evaluates
    NURBS exactly at all nodes.

    Parameters
    ----------
    nurbs       : Surface_NURBS
    output_path : output .npz path
    n_div       : mesh divisions per direction, default 48

    Returns
    -------
    Surface_P2
    """
    n_u = n_v = n_div
    h = 1.0 / n_div

    # Build mesh
    nodes_2d, triangles = _build_p2_mesh(n_u, n_v)
    n_nodes = nodes_2d.shape[0]
    n_elem = triangles.shape[0]

    # Evaluate NURBS at all P2 nodes
    nodes_3d = nurbs.eval(nodes_2d)

    # Save
    np.savez_compressed(
        Path(output_path),
        method=np.array(['P2']),
        degree=np.array([2]),
        n_u=np.array([n_u]),
        n_v=np.array([n_v]),
        mesh_h=np.array([h]),
        n_nodes=np.array([n_nodes]),
        n_elements=np.array([n_elem]),
        nodes_2d=nodes_2d,
        nodes_3d=nodes_3d,
        triangles=triangles,
    )

    return Surface_P2(
        nodes_2d=nodes_2d,
        nodes_3d=nodes_3d,
        triangles=triangles,
        n_u=n_u,
        n_v=n_v,
        h=h,
    )


# ============================================================================
# 9. Public API — Load P1 / P2 models from npz
# ============================================================================

def load_p1_from_npz(npz_path: str | Path) -> Surface_P1:
    """Restore P1 linear FEM discretization from .npz file.

    Parameters
    ----------
    npz_path : .npz file path

    Returns
    -------
    Surface_P1
    """
    data = np.load(npz_path, allow_pickle=False)

    return Surface_P1(
        nodes_2d=data['nodes_2d'],
        nodes_3d=data['nodes_3d'],
        triangles=data['triangles'],
        n_u=int(data['n_u'][0]),
        n_v=int(data['n_v'][0]),
        h=float(data['mesh_h'][0]),
    )


def load_p2_from_npz(npz_path: str | Path) -> Surface_P2:
    """Restore P2 quadratic FEM discretization from .npz file.

    Parameters
    ----------
    npz_path : .npz file path

    Returns
    -------
    Surface_P2
    """
    data = np.load(npz_path, allow_pickle=False)

    return Surface_P2(
        nodes_2d=data['nodes_2d'],
        nodes_3d=data['nodes_3d'],
        triangles=data['triangles'],
        n_u=int(data['n_u'][0]),
        n_v=int(data['n_v'][0]),
        h=float(data['mesh_h'][0]),
    )


# ============================================================================
# 10. Public API — Four-model 3D visualization
# ============================================================================

def visualize_models(
    truth: Surface_Truth | None = None,
    nurbs: Surface_NURBS | None = None,
    p1: Surface_P1 | None = None,
    p2: Surface_P2 | None = None,
    *,
    nurbs_eval_res: int = 80,
    elev: float = 25.0,
    azim: float = -45.0,
    show: bool = True,
    figsize: tuple = (18, 14),
    dpi: int = 120,
    save_path: str | Path | None = None,
) :
    """Display four surface representations simultaneously in a 2x2 panel.

    +--------------+--------------+
    | Surface_Truth| Surface_NURBS|
    | (analytic)   | (surface+ctl)|
    +--------------+--------------+
    |  Surface_P1  |  Surface_P2  |
    | (tri wire)   | (mesh+midpt) |
    +--------------+--------------+

    Parameters
    ----------
    truth : Surface_Truth | None — truth model, skip if None
    nurbs : Surface_NURBS | None — NURBS model, skip if None
    p1    : Surface_P1 | None    — P1 discretization, skip if None
    p2    : Surface_P2 | None    — P2 discretization, skip if None
    nurbs_eval_res : NURBS evaluation resolution, default 80
    elev, azim     : unified view angle for all subplots, default (25, -45)
    show           : whether to call plt.show(), default True
    figsize        : figure size, default (18, 14)
    dpi            : output resolution, default 120
    save_path      : optional save path (.png/.pdf), skip if None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    provided = [m for m in [truth, nurbs, p1, p2] if m is not None]
    if not provided:
        raise ValueError("At least one model must be provided (truth / nurbs / p1 / p2)")

    # ---------- Gather global geometric extent ----------
    all_pts = []

    # --- Prepare panel data ---
    panes = {}  # key → dict with title, plot_fn

    if truth is not None:
        x, y, z = truth.x, truth.y, truth.z
        all_pts.append(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1))
        # Color: based on perturbation field delta
        d = truth.delta
        norm_t = plt.Normalize(d.min(), d.max())
        fc_t = cm.viridis(norm_t(d))
        panes['truth'] = {
            'title': f'Surface_Truth \nR={truth.R}, r={truth.r}, {truth.n_u}×{truth.n_v}',
            'plot_fn': lambda ax: ax.plot_surface(
                x, y, z, facecolors=fc_t, rstride=1, cstride=1,
                alpha=0.92, linewidth=0, antialiased=True, shade=True,
                lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
            ),
        }

    if nurbs is not None:
        # Evaluate NURBS surface
        u_fine = np.linspace(0, 1, nurbs_eval_res)
        v_fine = np.linspace(0, 1, nurbs_eval_res)
        ug, vg = np.meshgrid(u_fine, v_fine, indexing='ij')
        uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)
        x_n = nurbs.eval(uv_flat).reshape(nurbs_eval_res, nurbs_eval_res, 3)
        all_pts.append(x_n.reshape(-1, 3))

        # Color: |z|/r
        rv = np.sqrt(x_n[:, :, 0]**2 + x_n[:, :, 1]**2)
        geo = np.abs(x_n[:, :, 2]) / (rv.clip(0.01) + 1e-6)
        norm_n = plt.Normalize(geo.min(), geo.max())
        fc_n = cm.viridis(norm_n(geo))

        # Control net wireframe
        cp = nurbs.control_points  # (cu, cv, 3)

        def _plot_nurbs(ax):
            ax.plot_surface(
                x_n[:, :, 0], x_n[:, :, 1], x_n[:, :, 2],
                facecolors=fc_n, rstride=1, cstride=1,
                alpha=0.88, linewidth=0, antialiased=True, shade=True,
                lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
            )
            # Control net (sparse)
            stride_c = max(1, nurbs.ctrl_u // 8)
            for j in range(0, nurbs.ctrl_v, stride_c):
                line = cp[:, j, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.6, alpha=0.6)
            for i in range(0, nurbs.ctrl_u, stride_c):
                line = cp[i, :, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.6, alpha=0.6)

        panes['nurbs'] = {
            'title': f'Surface_NURBS (deg={nurbs.degree_u},{nurbs.degree_v})\n'
                     f'ctrl=({nurbs.ctrl_u},{nurbs.ctrl_v}), eval={nurbs_eval_res}²',
            'plot_fn': _plot_nurbs,
        }

    if p1 is not None:
        # P1: sample piecewise linear surface on fine grid
        M = max(60, p1.n_u * 3)
        u_f = np.linspace(0, 1, M)
        v_f = np.linspace(0, 1, M)
        x_p1 = _eval_p1_on_fine_grid(u_f, v_f, p1.nodes_3d, p1.n_u, p1.n_v)
        all_pts.append(x_p1.reshape(-1, 3))

        # Color
        rv = np.sqrt(x_p1[:, :, 0]**2 + x_p1[:, :, 1]**2)
        geo = np.abs(x_p1[:, :, 2]) / (rv.clip(0.01) + 1e-6)
        norm_p1 = plt.Normalize(geo.min(), geo.max())
        fc_p1 = cm.viridis(norm_p1(geo))

        # Mesh wireframe
        verts = p1.nodes_3d.reshape(p1.n_u + 1, p1.n_v + 1, 3)

        def _plot_p1(ax):
            ax.plot_surface(
                x_p1[:, :, 0], x_p1[:, :, 1], x_p1[:, :, 2],
                facecolors=fc_p1, rstride=1, cstride=1,
                alpha=0.90, linewidth=0, antialiased=True, shade=True,
                lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
            )
            stride_m = max(1, p1.n_u // 6)
            for j in range(0, p1.n_v + 1, stride_m):
                line = verts[:, j, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)
            for i in range(0, p1.n_u + 1, stride_m):
                line = verts[i, :, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)

        panes['p1'] = {
            'title': f'Surface_P1 \n'
                     f'{p1.n_u}×{p1.n_v}, h={p1.h:.4f}, '
                     f'{p1.nodes_3d.shape[0]} nodes, {p1.triangles.shape[0]} elems',
            'plot_fn': _plot_p1,
        }

    if p2 is not None:
        M = max(60, p2.n_u * 3)
        u_f = np.linspace(0, 1, M)
        v_f = np.linspace(0, 1, M)
        x_p2 = _eval_p2_on_fine_grid(u_f, v_f, p2.nodes_3d, p2.n_u, p2.n_v)
        all_pts.append(x_p2.reshape(-1, 3))

        rv = np.sqrt(x_p2[:, :, 0]**2 + x_p2[:, :, 1]**2)
        geo = np.abs(x_p2[:, :, 2]) / (rv.clip(0.01) + 1e-6)
        norm_p2 = plt.Normalize(geo.min(), geo.max())
        fc_p2 = cm.viridis(norm_p2(geo))

        verts = p2.nodes_3d[:(p2.n_u + 1) * (p2.n_v + 1)].reshape(
            p2.n_u + 1, p2.n_v + 1, 3)

        def _plot_p2(ax):
            ax.plot_surface(
                x_p2[:, :, 0], x_p2[:, :, 1], x_p2[:, :, 2],
                facecolors=fc_p2, rstride=1, cstride=1,
                alpha=0.90, linewidth=0, antialiased=True, shade=True,
                lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
            )
            stride_m = max(1, p2.n_u // 6)
            for j in range(0, p2.n_v + 1, stride_m):
                line = verts[:, j, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)
            for i in range(0, p2.n_u + 1, stride_m):
                line = verts[i, :, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)

        panes['p2'] = {
            'title': f'Surface_P2 \n'
                     f'{p2.n_u}×{p2.n_v}, h={p2.h:.4f}, '
                     f'{p2.nodes_3d.shape[0]} nodes, {p2.triangles.shape[0]} elems',
            'plot_fn': _plot_p2,
        }

    # ---------- Global coordinate extent ----------
    all_pts = np.vstack(all_pts)
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    mid = all_pts.mean(axis=0)

    # ---------- 2x2 layout ----------
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        r'$\mathbf{Surface\ Model\ —\ Four\ Representations}$'
        '\nT² Torus with Multi-Frequency Sine Wave Perturbation',
        fontsize=15, fontweight='bold', y=0.98,
    )

    positions = {'truth': (0, 0), 'nurbs': (0, 1), 'p1': (1, 0), 'p2': (1, 1)}
    for key, (row, col) in positions.items():
        if key not in panes:
            continue
        ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection='3d')
        info = panes[key]
        info['plot_fn'](ax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(info['title'], fontsize=11, pad=8)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06,
                         wspace=0.10, hspace=0.28)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.3)

    if show:
        plt.show()

    return fig

def visualize_single_model(
    model: Surface_Truth | Surface_NURBS | Surface_P1 | Surface_P2,
    *,
    title: str | None = None,
    elev: float = 25.0,
    azim: float = -45.0,
    show: bool = True,
    figsize: tuple = (12, 10),
    dpi: int = 120,
    save_path: str | Path | None = None,
    n_samples: int = 120,
    show_control_net: bool = True,
    show_wireframe: bool = True,
    show_nodes: bool = False,
):
    """Convenient single-model visualization with automatic best rendering and adaptive bounding.

    Rendering strategies:
      Surface_Truth  -> high-res surface + perturbation color map
      Surface_NURBS  -> smooth surface + control net wireframe (toggleable)
      Surface_P1     -> triangular mesh + structured wireframe (toggleable)
      Surface_P2     -> triangular mesh + node labels (toggleable)

    Parameters
    ----------
    model : Surface_Truth | Surface_NURBS | Surface_P1 | Surface_P2
    title : Custom title, auto-generated if None
    elev, azim : View angle params
    show  : Whether to call plt.show()
    figsize : Figure size
    dpi   : Output resolution
    save_path : Optional save path
    n_samples : Surface evaluation resolution (for NURBS/P1/P2)
    show_control_net : Whether to show NURBS control net
    show_wireframe   : Whether to show P1/P2 structured wireframe
    show_nodes       : Whether to label P2 edge midpoints (large markers)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    pts_for_bbox = None  # Gather point cloud for adaptive bounding box

    # -- Surface_Truth --
    if isinstance(model, Surface_Truth):
        d = model.delta
        norm = plt.Normalize(d.min(), d.max())
        fc = cm.viridis(norm(d))
        ax.plot_surface(
            model.x, model.y, model.z,
            facecolors=fc, rstride=1, cstride=1,
            alpha=0.92, linewidth=0, antialiased=True, shade=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
        )
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cbar.set_label(r'$\delta(u,v)$', fontsize=10)
        
        pts_for_bbox = np.stack([model.x.ravel(), model.y.ravel(), model.z.ravel()], axis=-1)

        if title is None:
            title = (f'Surface_Truth \n'
                     f'R={model.R}, r={model.r}, {model.n_u}×{model.n_v}, '
                     f'{len(model.wave_components)} wave components')

    # -- Surface_NURBS --
    elif isinstance(model, Surface_NURBS):
        u_fine = np.linspace(0, 1, n_samples)
        v_fine = np.linspace(0, 1, n_samples)
        ug, vg = np.meshgrid(u_fine, v_fine, indexing='ij')
        uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)
        X = model.eval(uv_flat).reshape(n_samples, n_samples, 3)

        rv = np.sqrt(X[:, :, 0]**2 + X[:, :, 1]**2)
        geo = np.abs(X[:, :, 2]) / (rv.clip(0.01) + 1e-6)
        norm = plt.Normalize(geo.min(), geo.max())
        fc = cm.viridis(norm(geo))

        ax.plot_surface(
            X[:, :, 0], X[:, :, 1], X[:, :, 2],
            facecolors=fc, rstride=1, cstride=1,
            alpha=0.88, linewidth=0, antialiased=True, shade=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
        )
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cbar.set_label(r'$|z|/r$ (geometry)', fontsize=10)
        
        pts_for_bbox = X.reshape(-1, 3)

        if show_control_net:
            cp = model.control_points
            stride_c = max(1, model.ctrl_u // 8)
            for j in range(0, model.ctrl_v, stride_c):
                line = cp[:, j, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.6, alpha=0.6)
            for i in range(0, model.ctrl_u, stride_c):
                line = cp[i, :, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.6, alpha=0.6)

        if title is None:
            title = (f'Surface_NURBS (deg={model.degree_u},{model.degree_v})\n'
                     f'ctrl=({model.ctrl_u},{model.ctrl_v}), '
                     f'eval={n_samples}², R={model.torus_R}, r={model.torus_r}')

    # -- Surface_P1 --
    elif isinstance(model, Surface_P1):
        n_u, n_v = model.n_u, model.n_v
        M = max(60, n_u * 3)
        u_f = np.linspace(0, 1, M)
        v_f = np.linspace(0, 1, M)
        X = _eval_p1_on_fine_grid(u_f, v_f, model.nodes_3d, n_u, n_v)

        rv = np.sqrt(X[:, :, 0]**2 + X[:, :, 1]**2)
        geo = np.abs(X[:, :, 2]) / (rv.clip(0.01) + 1e-6)
        norm = plt.Normalize(geo.min(), geo.max())
        fc = cm.viridis(norm(geo))

        ax.plot_surface(
            X[:, :, 0], X[:, :, 1], X[:, :, 2],
            facecolors=fc, rstride=1, cstride=1,
            alpha=0.90, linewidth=0, antialiased=True, shade=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
        )
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cbar.set_label(r'$|z|/r$ (geometry)', fontsize=10)
        
        pts_for_bbox = X.reshape(-1, 3)

        if show_wireframe:
            verts = model.nodes_3d.reshape(n_u + 1, n_v + 1, 3)
            stride_m = max(1, n_u // 6)
            for j in range(0, n_v + 1, stride_m):
                line = verts[:, j, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)
            for i in range(0, n_u + 1, stride_m):
                line = verts[i, :, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)

        if title is None:
            title = (f'Surface_P1 \n'
                     f'{n_u}×{n_v}, h={model.h:.4f}, '
                     f'{model.nodes_3d.shape[0]} nodes, '
                     f'{model.triangles.shape[0]} elems')

    # -- Surface_P2 --
    elif isinstance(model, Surface_P2):
        n_u, n_v = model.n_u, model.n_v
        M = max(60, n_u * 3)
        u_f = np.linspace(0, 1, M)
        v_f = np.linspace(0, 1, M)
        X = _eval_p2_on_fine_grid(u_f, v_f, model.nodes_3d, n_u, n_v)

        rv = np.sqrt(X[:, :, 0]**2 + X[:, :, 1]**2)
        geo = np.abs(X[:, :, 2]) / (rv.clip(0.01) + 1e-6)
        norm = plt.Normalize(geo.min(), geo.max())
        fc = cm.viridis(norm(geo))

        ax.plot_surface(
            X[:, :, 0], X[:, :, 1], X[:, :, 2],
            facecolors=fc, rstride=1, cstride=1,
            alpha=0.90, linewidth=0, antialiased=True, shade=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
        )
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cbar.set_label(r'$|z|/r$ (geometry)', fontsize=10)
        
        pts_for_bbox = X.reshape(-1, 3)

        if show_wireframe:
            verts = model.nodes_3d[:(n_u + 1) * (n_v + 1)].reshape(
                n_u + 1, n_v + 1, 3)
            stride_m = max(1, n_u // 6)
            for j in range(0, n_v + 1, stride_m):
                line = verts[:, j, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)
            for i in range(0, n_u + 1, stride_m):
                line = verts[i, :, :]
                ax.plot(line[:, 0], line[:, 1], line[:, 2],
                        color='white', linewidth=0.5, alpha=0.5)

        if show_nodes:
            n_vert = (n_u + 1) * (n_v + 1)
            edge_mid = model.nodes_3d[n_vert:]  # All non-vertex nodes
            ax.scatter(
                edge_mid[:, 0], edge_mid[:, 1], edge_mid[:, 2],
                c='red', s=8, alpha=0.7, marker='o', label='Edge midpoints',
            )

        if title is None:
            title = (f'Surface_P2 \n'
                     f'{n_u}×{n_v}, h={model.h:.4f}, '
                     f'{model.nodes_3d.shape[0]} nodes, '
                     f'{model.triangles.shape[0]} elems')

    else:
        raise TypeError(
            f'Unsupported model type: {type(model).__name__}, '
            f'expected Surface_Truth | Surface_NURBS | Surface_P1 | Surface_P2'
        )

    # -- Adaptive surface sizing: compute cubic bounding box from all branch points --
    if pts_for_bbox is not None:
        max_range = np.ptp(pts_for_bbox, axis=0).max() / 2.0
        mid_x = (pts_for_bbox[:, 0].max() + pts_for_bbox[:, 0].min()) * 0.5
        mid_y = (pts_for_bbox[:, 1].max() + pts_for_bbox[:, 1].min()) * 0.5
        mid_z = (pts_for_bbox[:, 2].max() + pts_for_bbox[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    # Equal aspect ratio ensures consistent scaling across all axes
    ax.set_aspect('equal')

    if show_nodes and isinstance(model, Surface_P2):
        ax.legend(fontsize=9, loc='upper right')

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.3)

    if show:
        plt.show()

    return fig

def visualize_trimmed_nurbs(
    nurbs: Surface_NURBS,
    trim_loop: np.ndarray,
    *,
    title: str | None = None,
    elev: float = 25.0,
    azim: float = -45.0,
    show: bool = True,
    figsize: tuple = (12, 10),
    dpi: int = 120,
    save_path: str | Path | None = None,
    n_samples: int = 150,
    show_control_net: bool = True,
):
    """Visualize NURBS surface with trim boundary, showing only the actual patch (ignoring underlying surface extensions).

    Parameters
    ----------
    nurbs : Surface_NURBS
    trim_loop : (N, 2) trim boundary points in [0,1]^2 parametric domain (closed polygon)
    title, elev, azim, show, figsize, dpi, save_path : Same as visualize_single_model
    n_samples : Parametric domain sampling resolution, larger = smoother boundary
    show_control_net : Whether to show control net

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.path import Path
    from matplotlib.tri import Triangulation

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 1. Generate structured grid on [0,1]^2
    u_fine = np.linspace(0, 1, n_samples)
    v_fine = np.linspace(0, 1, n_samples)
    ug, vg = np.meshgrid(u_fine, v_fine, indexing='ij')
    uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)

    # 2. Check whether each parameter point is inside trim boundary
    trim_path = Path(trim_loop)
    inside = trim_path.contains_points(uv_flat)

    # 3. Evaluate NURBS at all parameter points
    X = nurbs.eval(uv_flat)

    # 4. Build structured triangular mesh, discard triangles intersecting exterior
    n_u = len(u_fine)
    n_v = len(v_fine)
    triangles = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            v00 = i * n_v + j
            v10 = (i + 1) * n_v + j
            v11 = (i + 1) * n_v + (j + 1)
            v01 = i * n_v + (j + 1)
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    triangles = np.array(triangles)

    # Remove triangle if any vertex is outside boundary
    tri_mask = np.any(~inside[triangles], axis=1)
    visible_triangles = triangles[~tri_mask]

    # 5. Color: reuse |z|/r geometric color mapping
    rv = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    geo = np.abs(X[:, 2]) / (rv.clip(0.01) + 1e-6)
    norm = plt.Normalize(geo.min(), geo.max())

    # 6. Plot trimmed triangular surface
    #    Color array passed as positional arg (4th param), plot_trisurf auto-interpolates per vertex.
    #    Note: cannot use C= keyword (errors with Poly3DCollection), nor set_array
    #    (Poly3DCollection's array is per-face, length mismatch).
    ax.plot_trisurf(
        X[:, 0], X[:, 1], X[:, 2], geo,
        triangles=visible_triangles,
        cmap='viridis',
        alpha=0.90,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
    cbar.set_label(r'$|z|/r$ (geometry)', fontsize=10)

    # 7. Control net (only show control points inside trim region, avoid visual clutter)
    if show_control_net:
        cp = nurbs.control_points  # (cu, cv, 3)
        # Compute normalized parametric coordinates for control points (i/(cu-1), j/(cv-1))
        i_idx = np.arange(nurbs.ctrl_u)
        j_idx = np.arange(nurbs.ctrl_v)
        u_cp = i_idx / max(1, nurbs.ctrl_u - 1)
        v_cp = j_idx / max(1, nurbs.ctrl_v - 1)
        ug_cp, vg_cp = np.meshgrid(u_cp, v_cp, indexing='ij')
        uv_cp = np.stack([ug_cp.ravel(), vg_cp.ravel()], axis=-1)
        cp_visible = trim_path.contains_points(uv_cp).reshape(nurbs.ctrl_u, nurbs.ctrl_v)

        stride_c = max(1, nurbs.ctrl_u // 8)
        for j in range(0, nurbs.ctrl_v, stride_c):
            line = cp[:, j, :]
            vis = cp_visible[:, j]
            # Only plot continuous visible segments
            _plot_visible_segments(ax, line, vis)
        for i in range(0, nurbs.ctrl_u, stride_c):
            line = cp[i, :, :]
            vis = cp_visible[i, :]
            _plot_visible_segments(ax, line, vis)

    if title is None:
        title = (f'Surface_NURBS (deg={nurbs.degree_u},{nurbs.degree_v})\n'
                 f'ctrl=({nurbs.ctrl_u},{nurbs.ctrl_v}), '
                 f'eval={n_samples}², R={nurbs.torus_R}, r={nurbs.torus_r}')

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_aspect('equal')

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.3)

    if show:
        plt.show()

    return fig


def _plot_visible_segments(ax, line_pts, visibility):
    """Helper: split control point polyline into visible segments by visibility mask."""
    n = len(line_pts)
    start = 0
    while start < n:
        # Find next visible segment start
        while start < n and not visibility[start]:
            start += 1
        if start >= n:
            break
        end = start
        while end < n and visibility[end]:
            end += 1
        segment = line_pts[start:end]
        if len(segment) > 1:
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                    color='white', linewidth=0.6, alpha=0.6)
        start = end


# ============================================================================
# 11. Internal: P1 / P2 fine-grid evaluation (for visualization)
# ============================================================================

def _eval_p1_on_fine_grid(u_fine, v_fine, nodes_3d, n_u, n_v):
    """Evaluate P1 piecewise linear approximation on fine (u, v) grid."""
    u_grid, v_grid = np.meshgrid(u_fine, v_fine, indexing='ij')
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()

    cell_i = np.clip((u_flat * n_u).astype(int), 0, n_u - 1)
    cell_j = np.clip((v_flat * n_v).astype(int), 0, n_v - 1)

    du = 1.0 / n_u
    dv = 1.0 / n_v
    xi  = (u_flat - cell_i * du) / du
    eta = (v_flat - cell_j * dv) / dv

    in_tri1 = xi >= eta

    la = np.where(in_tri1, 1.0 - xi,  1.0 - eta)
    lb = np.where(in_tri1, xi - eta,  xi)
    lc = np.where(in_tri1, eta,       eta - xi)

    idx_v00 = cell_j       + cell_i       * (n_v + 1)
    idx_v10 = cell_j       + (cell_i + 1) * (n_v + 1)
    idx_v11 = (cell_j + 1) + (cell_i + 1) * (n_v + 1)
    idx_v01 = (cell_j + 1) + cell_i       * (n_v + 1)

    idx1 = idx_v00
    idx2 = np.where(in_tri1, idx_v10, idx_v11)
    idx3 = np.where(in_tri1, idx_v11, idx_v01)

    result = (la[:, None] * nodes_3d[idx1] +
              lb[:, None] * nodes_3d[idx2] +
              lc[:, None] * nodes_3d[idx3])

    M = len(u_fine)
    return result.reshape(M, M, 3)


def _eval_p2_on_fine_grid(u_fine, v_fine, nodes_3d, n_u, n_v):
    """Evaluate P2 piecewise quadratic approximation on fine (u, v) grid."""
    u_grid, v_grid = np.meshgrid(u_fine, v_fine, indexing='ij')
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()

    cell_i = np.clip((u_flat * n_u).astype(int), 0, n_u - 1)
    cell_j = np.clip((v_flat * n_v).astype(int), 0, n_v - 1)

    du = 1.0 / n_u
    dv = 1.0 / n_v
    xi  = (u_flat - cell_i * du) / du
    eta = (v_flat - cell_j * dv) / dv

    in_tri1 = xi >= eta

    la = np.where(in_tri1, 1.0 - xi,  1.0 - eta)
    lb = np.where(in_tri1, xi - eta,  xi)
    lc = np.where(in_tri1, eta,       eta - xi)

    N1 = la * (2.0 * la - 1.0)
    N2 = lb * (2.0 * lb - 1.0)
    N3 = lc * (2.0 * lc - 1.0)
    N4 = 4.0 * la * lb
    N5 = 4.0 * lb * lc
    N6 = 4.0 * la * lc

    n_vert   = (n_u + 1) * (n_v + 1)
    n_h      = n_u * (n_v + 1)
    n_v_edge = (n_u + 1) * n_v

    idx_v00 = cell_j       + cell_i       * (n_v + 1)
    idx_v10 = cell_j       + (cell_i + 1) * (n_v + 1)
    idx_v11 = (cell_j + 1) + (cell_i + 1) * (n_v + 1)
    idx_v01 = (cell_j + 1) + cell_i       * (n_v + 1)

    idx_h00   = n_vert + cell_j       + cell_i       * (n_v + 1)
    idx_h01   = n_vert + (cell_j + 1) + cell_i       * (n_v + 1)
    idx_v10_e = n_vert + n_h + cell_j + (cell_i + 1) * n_v
    idx_v00_e = n_vert + n_h + cell_j + cell_i       * n_v
    idx_d00   = n_vert + n_h + n_v_edge + cell_j + cell_i * n_v

    idx1 = idx_v00
    idx2 = np.where(in_tri1, idx_v10, idx_v11)
    idx3 = np.where(in_tri1, idx_v11, idx_v01)
    idx4 = np.where(in_tri1, idx_h00,   idx_d00)
    idx5 = np.where(in_tri1, idx_v10_e, idx_h01)
    idx6 = np.where(in_tri1, idx_d00,   idx_v00_e)

    result = (N1[:, None] * nodes_3d[idx1] +
              N2[:, None] * nodes_3d[idx2] +
              N3[:, None] * nodes_3d[idx3] +
              N4[:, None] * nodes_3d[idx4] +
              N5[:, None] * nodes_3d[idx5] +
              N6[:, None] * nodes_3d[idx6])

    M = len(u_fine)
    return result.reshape(M, M, 3)



# ============================================================================
# 13. Public API — Geometry error computation (param-independent, point-cloud based)
# ============================================================================
# Note: geomdl NURBS fitting uses chord-length parameterization, so there is
# no simple linear correspondence between NURBS (u,v) in [0,1]^2 and
# Truth (2pi*u, 2pi*v) in [0, 2pi)^2. All error computations use point-cloud
# comparison: dense sampling on both model and truth surfaces, then KD-Tree
# nearest-neighbor search for Chamfer and Hausdorff distances.

from scipy.spatial import cKDTree


def _eval_truth_at_arbitrary_points(truth: Surface_Truth,
                                     uv: np.ndarray) -> np.ndarray:
    """Evaluate truth surface (analytic formula) at arbitrary (u,v) in [0, 2pi)^2.

    Parameters
    ----------
    truth : Surface_Truth
    uv    : (N, 2) — (u, v) parameter coordinates, domain [0, 2pi)^2

    Returns
    -------
    pts : (N, 3) — 3D points on surface
    """
    u_arr = uv[:, 0]
    v_arr = uv[:, 1]

    delta = np.zeros_like(u_arr)
    for amp, fu, fv, pu, pv in truth.wave_components:
        delta += amp * np.sin(fu * u_arr + pu) * np.sin(fv * v_arr + pv)

    r_eff = truth.r + delta
    x = (truth.R + r_eff * np.cos(v_arr)) * np.cos(u_arr)
    y = (truth.R + r_eff * np.cos(v_arr)) * np.sin(u_arr)
    z = r_eff * np.sin(v_arr)

    return np.stack([x, y, z], axis=-1)


def _eval_p1_at_arbitrary_points(uv: np.ndarray,
                                  p1: Surface_P1) -> np.ndarray:
    """Evaluate P1 piecewise linear interpolation at arbitrary (u,v) in [0, 1]^2.

    Uses barycentric coordinates to locate triangle element and perform linear interpolation.

    Parameters
    ----------
    uv : (N, 2) — parameter coordinates
    p1 : Surface_P1

    Returns
    -------
    pts : (N, 3) — interpolated 3D points
    """
    n_u, n_v = p1.n_u, p1.n_v
    u_arr = uv[:, 0]
    v_arr = uv[:, 1]
    du = 1.0 / n_u
    dv = 1.0 / n_v

    cell_i = np.clip((u_arr / du).astype(int), 0, n_u - 1)
    cell_j = np.clip((v_arr / dv).astype(int), 0, n_v - 1)

    xi = (u_arr - cell_i * du) / du
    eta = (v_arr - cell_j * dv) / dv

    in_tri1 = xi >= eta

    # Barycentric coordinates
    la = np.where(in_tri1, 1.0 - xi, 1.0 - eta)
    lb = np.where(in_tri1, xi - eta, xi)
    lc = np.where(in_tri1, eta, eta - xi)

    # Triangle cell vertex indices
    idx_v00 = cell_j + cell_i * (n_v + 1)
    idx_v10 = cell_j + (cell_i + 1) * (n_v + 1)
    idx_v11 = (cell_j + 1) + (cell_i + 1) * (n_v + 1)
    idx_v01 = (cell_j + 1) + cell_i * (n_v + 1)

    idx1 = idx_v00
    idx2 = np.where(in_tri1, idx_v10, idx_v11)
    idx3 = np.where(in_tri1, idx_v11, idx_v01)

    result = (la[:, None] * p1.nodes_3d[idx1] +
              lb[:, None] * p1.nodes_3d[idx2] +
              lc[:, None] * p1.nodes_3d[idx3])
    return result


def _eval_p2_at_arbitrary_points(uv: np.ndarray,
                                  p2: Surface_P2) -> np.ndarray:
    """Evaluate P2 piecewise quadratic interpolation at arbitrary (u,v) in [0, 1]^2.

    Uses 6-node Lagrange quadratic basis functions.

    Parameters
    ----------
    uv : (N, 2) — parameter coordinates
    p2 : Surface_P2

    Returns
    -------
    pts : (N, 3) — interpolated 3D points
    """
    n_u, n_v = p2.n_u, p2.n_v
    u_arr = uv[:, 0]
    v_arr = uv[:, 1]
    du = 1.0 / n_u
    dv = 1.0 / n_v

    cell_i = np.clip((u_arr / du).astype(int), 0, n_u - 1)
    cell_j = np.clip((v_arr / dv).astype(int), 0, n_v - 1)

    xi = (u_arr - cell_i * du) / du
    eta = (v_arr - cell_j * dv) / dv

    in_tri1 = xi >= eta

    la = np.where(in_tri1, 1.0 - xi, 1.0 - eta)
    lb = np.where(in_tri1, xi - eta, xi)
    lc = np.where(in_tri1, eta, eta - xi)

    # P2 6-node Lagrange basis functions
    N1 = la * (2.0 * la - 1.0)
    N2 = lb * (2.0 * lb - 1.0)
    N3 = lc * (2.0 * lc - 1.0)
    N4 = 4.0 * la * lb
    N5 = 4.0 * lb * lc
    N6 = 4.0 * la * lc

    n_vert = (n_u + 1) * (n_v + 1)
    n_h = n_u * (n_v + 1)
    n_v_edge = (n_u + 1) * n_v

    # 6-node indices per 2x2 cell
    idx_v00 = cell_j + cell_i * (n_v + 1)
    idx_v10 = cell_j + (cell_i + 1) * (n_v + 1)
    idx_v11 = (cell_j + 1) + (cell_i + 1) * (n_v + 1)
    idx_v01 = (cell_j + 1) + cell_i * (n_v + 1)

    idx_h00 = n_vert + cell_j + cell_i * (n_v + 1)
    idx_h01 = n_vert + (cell_j + 1) + cell_i * (n_v + 1)
    idx_v10_e = n_vert + n_h + cell_j + (cell_i + 1) * n_v
    idx_v00_e = n_vert + n_h + cell_j + cell_i * n_v
    idx_d00 = n_vert + n_h + n_v_edge + cell_j + cell_i * n_v

    idx1 = idx_v00
    idx2 = np.where(in_tri1, idx_v10, idx_v11)
    idx3 = np.where(in_tri1, idx_v11, idx_v01)
    idx4 = np.where(in_tri1, idx_h00, idx_d00)
    idx5 = np.where(in_tri1, idx_v10_e, idx_h01)
    idx6 = np.where(in_tri1, idx_d00, idx_v00_e)

    result = (N1[:, None] * p2.nodes_3d[idx1] +
              N2[:, None] * p2.nodes_3d[idx2] +
              N3[:, None] * p2.nodes_3d[idx3] +
              N4[:, None] * p2.nodes_3d[idx4] +
              N5[:, None] * p2.nodes_3d[idx5] +
              N6[:, None] * p2.nodes_3d[idx6])
    return result


def _sample_truth_point_cloud(truth: Surface_Truth,
                               n_samples: int) -> np.ndarray:
    """Dense sampling of Surface_Truth surface, returns (N, 3) point cloud.

    Uniformly sampled on own parameter domain [0, 2pi)^2.
    """
    u = np.linspace(0, 2 * np.pi, n_samples)
    v = np.linspace(0, 2 * np.pi, n_samples)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)
    return _eval_truth_at_arbitrary_points(truth, uv_flat)


def _sample_nurbs_point_cloud(nurbs: Surface_NURBS,
                               n_samples: int) -> np.ndarray:
    """Dense sampling of Surface_NURBS surface, returns (N, 3) point cloud.

    Uniformly sampled on own parameter domain [0, 1]^2.
    """
    u = np.linspace(0, 1, n_samples)
    v = np.linspace(0, 1, n_samples)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)
    return nurbs.eval(uv_flat)


def _sample_p1_point_cloud(p1: Surface_P1, n_samples: int) -> np.ndarray:
    """Dense sampling of Surface_P1 surface, returns (N, 3) point cloud.

    Uniformly sampled on own parameter domain [0, 1]^2, using piecewise linear interpolation.
    """
    u = np.linspace(0, 1, n_samples)
    v = np.linspace(0, 1, n_samples)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)
    return _eval_p1_at_arbitrary_points(uv_flat, p1)


def _sample_p2_point_cloud(p2: Surface_P2, n_samples: int) -> np.ndarray:
    """Dense sampling of Surface_P2 surface, returns (N, 3) point cloud.

    Uniformly sampled on own parameter domain [0, 1]^2, using piecewise quadratic interpolation.
    """
    u = np.linspace(0, 1, n_samples)
    v = np.linspace(0, 1, n_samples)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)
    return _eval_p2_at_arbitrary_points(uv_flat, p2)


def _compute_point_cloud_errors(pts_model: np.ndarray,
                                 pts_truth: np.ndarray) -> dict:
    """Given two point clouds, compute parameterization-independent geometric error using KD-Tree.

    Parameters
    ----------
    pts_model : (M, 3) — model surface sample point cloud
    pts_truth : (N, 3) — truth surface sample point cloud

    Returns
    -------
    dict with keys:
        'chamfer'       — Chamfer distance: (mean(dist_M→T) + mean(dist_T→M)) / 2
        'hausdorff'     — Hausdorff distance: max( max(dist_M→T), max(dist_T→M) )
        'mean_model_to_truth' — model → Truth mean nearest distance
        'mean_truth_to_model' — Truth → modelmean nearest distance
        'max_model_to_truth'  — model → Truth max nearest distance
        'max_truth_to_model'  — Truth → modelmax nearest distance
        'rms_model_to_truth'  — model → Truth RMS nearest distance
    """
    tree_truth = cKDTree(pts_truth)
    tree_model = cKDTree(pts_model)

    dist_m2t, _ = tree_truth.query(pts_model, k=1)
    dist_t2m, _ = tree_model.query(pts_truth, k=1)

    chamfer = float((np.mean(dist_m2t) + np.mean(dist_t2m)) / 2.0)
    hausdorff = float(max(np.max(dist_m2t), np.max(dist_t2m)))

    return {
        'chamfer': chamfer,
        'hausdorff': hausdorff,
        'mean_model_to_truth': float(np.mean(dist_m2t)),
        'mean_truth_to_model': float(np.mean(dist_t2m)),
        'max_model_to_truth': float(np.max(dist_m2t)),
        'max_truth_to_model': float(np.max(dist_t2m)),
        'rms_model_to_truth': float(np.sqrt(np.mean(dist_m2t ** 2))),
    }


def compute_geometry_error_nurbs(nurbs: Surface_NURBS,
                                  truth: Surface_Truth,
                                  n_samples: int = 200) -> dict:
    """Compute geometry error between Surface_NURBS and Surface_Truth (parameterization-independent).

    Densely sample both surfaces at n_samples x n_samples,
    compute Chamfer and Hausdorff distances via KD-Tree nearest-neighbor search.

    Parameters
    ----------
    nurbs     : Surface_NURBS
    truth     : Surface_Truth
    n_samples : Samples per direction, default 200

    Returns
    -------
    dict with keys:
        'chamfer'    — Chamfer distance
        'hausdorff'  — Hausdorff distance (max one-sided nearest distance)
        'rms'        — model → Truth RMS nearest distance
        'mean'       — model → Truth mean nearest distance
        'n_samples'  — samplingresolution
    """
    pts_nurbs = _sample_nurbs_point_cloud(nurbs, n_samples)
    pts_truth = _sample_truth_point_cloud(truth, n_samples)

    result = _compute_point_cloud_errors(pts_nurbs, pts_truth)
    result['n_samples'] = n_samples
    result['n_pts_model'] = pts_nurbs.shape[0]
    result['n_pts_truth'] = pts_truth.shape[0]

    # Backward-compatible key aliases
    truth_rms_norm = float(np.sqrt(np.mean(np.sum(pts_truth ** 2, axis=1))))
    result['L2_rel'] = result['rms_model_to_truth'] / truth_rms_norm if truth_rms_norm > 0 else 0.0
    result['L2_abs'] = result['rms_model_to_truth']
    result['Linf'] = result['hausdorff']
    result['mean_err'] = result['mean_model_to_truth']
    return result


def compute_geometry_error_p1(p1: Surface_P1,
                               truth: Surface_Truth,
                               n_samples: int = 200) -> dict:
    """Compute geometry error between Surface_P1 and Surface_Truth (parameterization-independent).

    Densely sample both surfaces at n_samples x n_samples,
    compute Chamfer and Hausdorff distances via KD-Tree nearest-neighbor search.

    Parameters
    ----------
    p1        : Surface_P1
    truth     : Surface_Truth
    n_samples : Samples per direction, default 200

    Returns
    -------
    Same dict as compute_geometry_error_nurbs
    """
    pts_p1 = _sample_p1_point_cloud(p1, n_samples)
    pts_truth = _sample_truth_point_cloud(truth, n_samples)

    result = _compute_point_cloud_errors(pts_p1, pts_truth)
    result['n_samples'] = n_samples
    result['n_pts_model'] = pts_p1.shape[0]
    result['n_pts_truth'] = pts_truth.shape[0]

    truth_rms_norm = float(np.sqrt(np.mean(np.sum(pts_truth ** 2, axis=1))))
    result['L2_rel'] = result['rms_model_to_truth'] / truth_rms_norm if truth_rms_norm > 0 else 0.0
    result['L2_abs'] = result['rms_model_to_truth']
    result['Linf'] = result['hausdorff']
    result['mean_err'] = result['mean_model_to_truth']
    return result


def compute_geometry_error_p2(p2: Surface_P2,
                               truth: Surface_Truth,
                               n_samples: int = 200) -> dict:
    """Compute geometry error between Surface_P2 and Surface_Truth (parameterization-independent).

    Densely sample both surfaces at n_samples x n_samples,
    compute Chamfer and Hausdorff distances via KD-Tree nearest-neighbor search.

    Parameters
    ----------
    p2        : Surface_P2
    truth     : Surface_Truth
    n_samples : Samples per direction, default 200

    Returns
    -------
    Same dict as compute_geometry_error_nurbs
    """
    pts_p2 = _sample_p2_point_cloud(p2, n_samples)
    pts_truth = _sample_truth_point_cloud(truth, n_samples)

    result = _compute_point_cloud_errors(pts_p2, pts_truth)
    result['n_samples'] = n_samples
    result['n_pts_model'] = pts_p2.shape[0]
    result['n_pts_truth'] = pts_truth.shape[0]

    truth_rms_norm = float(np.sqrt(np.mean(np.sum(pts_truth ** 2, axis=1))))
    result['L2_rel'] = result['rms_model_to_truth'] / truth_rms_norm if truth_rms_norm > 0 else 0.0
    result['L2_abs'] = result['rms_model_to_truth']
    result['Linf'] = result['hausdorff']
    result['mean_err'] = result['mean_model_to_truth']
    return result


def compute_geometry_error(
    model: Surface_NURBS | Surface_P1 | Surface_P2,
    truth: Surface_Truth,
    n_samples: int = 200,
) -> dict:
    """Unified error interface: auto-detect model type and compute geometry error vs Truth.

    Uses point-cloud comparison (parameterization-independent): dense sampling on model and Truth,
    compute Chamfer and Hausdorff distances via KD-Tree nearest-neighbor search.

    Parameters
    ----------
    model     : Surface_NURBS | Surface_P1 | Surface_P2
    truth     : Surface_Truth — Reference truth surface
    n_samples : Samples per direction, default 200

    Returns
    -------
    dict with:
        'chamfer'      — Chamfer distance
        'hausdorff'    — Hausdorff distance
        'rms' (L2_abs) — model→Truth RMS nearest distance
        'mean'         — model→Truth mean nearest distance
        'n_samples'    — samplingresolution
    """
    if isinstance(model, Surface_NURBS):
        return compute_geometry_error_nurbs(model, truth, n_samples)
    elif isinstance(model, Surface_P1):
        return compute_geometry_error_p1(model, truth, n_samples)
    elif isinstance(model, Surface_P2):
        return compute_geometry_error_p2(model, truth, n_samples)
    else:
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}, "
            f"expected Surface_NURBS | Surface_P1 | Surface_P2"
        )

def compute_mapping_error_vs_nurbs(
    model: Surface_P1 | Surface_P2,
    nurbs: Surface_NURBS,
    n_samples: int = 200
) -> dict:
    """Compute point-to-point mapping error between P1/P2 model and NURBS model on shared parametric domain [0,1]^2.
 
    This reflects the true FEM geometric mapping error (the root cause of metric tensor G deviation).
 
    Parameters
    ----------
    model     : Surface_P1 | Surface_P2
    nurbs     : Surface_NURBS
    n_samples : Samples per direction
 
    Returns
    -------
    dict with:
        'dist'                : (N,) np.ndarray — point-to-point Euclidean distance array at shared parameter points
        'chamfer'             : float — based on point-to-point distance Chamfer distance (At this point, theequivalent to bidirectional mean)
        'hausdorff'           : float — based on point-to-point distance Hausdorff distance (i.e. the maximum)
        'rms_model_to_truth'  : float — point-to-point mappingRMS distance (equivalent to L2_abs)
        'L2_rel'              : float — relative L2 error (RMS / reference surface RMS norm)
        'mean_model_to_truth' : float — meanpoint-to-point distance
    """
    u = np.linspace(0, 1, n_samples)
    v = np.linspace(0, 1, n_samples)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)
 
    # Evaluate at same parameter points
    if isinstance(model, Surface_P1):
        pts_model = _eval_p1_at_arbitrary_points(uv_flat, model)
    elif isinstance(model, Surface_P2):
        pts_model = _eval_p2_at_arbitrary_points(uv_flat, model)
    else:
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}, "
            f"expected Surface_P1 | Surface_P2"
        )
 
    pts_nurbs = nurbs.eval(uv_flat)
 
    # Compute point-to-point Euclidean distance
    dist = np.linalg.norm(pts_model - pts_nurbs, axis=1)
    
    nurbs_rms_norm = float(np.sqrt(np.mean(np.sum(pts_nurbs ** 2, axis=1))))
    
    rms_m2t = float(np.sqrt(np.mean(dist ** 2)))
    max_m2t = float(np.max(dist))
    mean_m2t = float(np.mean(dist))
    l2_rel = rms_m2t / nurbs_rms_norm if nurbs_rms_norm > 0 else 0.0
 
    return {
        'dist': dist,
        'chamfer': mean_m2t,                # Under strict point-to-point mapping, Chamfer reduces to mean distance
        'hausdorff': max_m2t,               # Under strict point-to-point mapping, Hausdorff reduces to max distance
        'rms_model_to_truth': rms_m2t,
        'L2_rel': l2_rel,
        'mean_model_to_truth': mean_m2t
    }



def refine_nurbs_h(nurbs: Surface_NURBS, target_ctrl_u: int, target_ctrl_v: int):
    """Perform h-refinement (knot insertion) on NURBS surface, pure matrix implementation.
    
    Builds Boehm knot insertion transfer matrices with pure NumPy, then uses tensor contraction
    to compute new control points exactly in 4D homogeneous space, avoiding dimension loss from black-box libraries.
    """
    import heapq

    diff_u = target_ctrl_u - nurbs.ctrl_u
    diff_v = target_ctrl_v - nurbs.ctrl_v


    if diff_u < 0 or diff_v < 0:
        raise ValueError(f"Target control point dimensions must be >= current dimensions.")
    if diff_u == 0 and diff_v == 0:
        print("error")
        return nurbs, np.eye(nurbs.ctrl_u), np.eye(nurbs.ctrl_v)

    def _get_insertion_knots(U, num_insertions):
        """Greedy bisection: always find the longest non-zero knot interval and split at midpoint"""
        if num_insertions <= 0:
            return []
        U_unique = np.unique(U)
        pq = []
        for i in range(len(U_unique) - 1):
            length = U_unique[i+1] - U_unique[i]
            heapq.heappush(pq, (-length, U_unique[i], U_unique[i+1]))
        
        insertions = []
        for _ in range(num_insertions):
            neg_l, start, end = heapq.heappop(pq)
            length = -neg_l
            mid = start + length / 2.0
            insertions.append(mid)
            # Push two new sub-intervals back into heap
            heapq.heappush(pq, (-(length / 2.0), start, mid))
            heapq.heappush(pq, (-(length / 2.0), mid, end))
            
        return sorted(insertions)

    def _insert_single_knot_matrix(U, p, u_bar):
        """Core: construct single knot insertion transfer matrix T"""
        k = np.searchsorted(U, u_bar, side='right') - 1
        if k >= len(U) - p - 1:
            k = len(U) - p - 2
        
        n = len(U) - p - 1  # Original control point count
        T = np.zeros((n + 1, n))
        
        for i in range(n + 1):
            if i <= k - p:
                alpha = 1.0
            elif i >= k + 1:
                alpha = 0.0
            else:
                denominator = U[i + p] - U[i]
                alpha = (u_bar - U[i]) / denominator if denominator != 0 else 0.0
            
            if i < n:
                T[i, i] = alpha
            if i - 1 >= 0:
                T[i, i - 1] += (1.0 - alpha)
                
        U_new = np.insert(U, k + 1, u_bar)
        return T, U_new

    # 1. Compute knots to insert
    knots_u = _get_insertion_knots(nurbs.knotvector_u, diff_u)
    knots_v = _get_insertion_knots(nurbs.knotvector_v, diff_v)

    # 2. Accumulate product for U-direction global transfer matrix (Shape: target_ctrl_u x current_ctrl_u)
    T_u = np.eye(nurbs.ctrl_u)
    U_new = nurbs.knotvector_u.copy()
    for ku in knots_u:
        T_step, U_new = _insert_single_knot_matrix(U_new, nurbs.degree_u, ku)
        T_u = T_step @ T_u 

    # 3. Accumulate product for V-direction global transfer matrix (Shape: target_ctrl_v x current_ctrl_v)
    T_v = np.eye(nurbs.ctrl_v)
    V_new = nurbs.knotvector_v.copy()
    for kv in knots_v:
        T_step, V_new = _insert_single_knot_matrix(V_new, nurbs.degree_v, kv)
        T_v = T_step @ T_v

    # 4. Lift 3D control points to 4D homogeneous space P^w
    # P_w = (x*w, y*w, z*w, w)
    Pw = np.empty((nurbs.ctrl_u, nurbs.ctrl_v, 4))
    W = nurbs.weights
    Pw[..., 0] = nurbs.control_points[..., 0] * W
    Pw[..., 1] = nurbs.control_points[..., 1] * W
    Pw[..., 2] = nurbs.control_points[..., 2] * W
    Pw[..., 3] = W

    # 5. Tensor contraction to compute new homogeneous control points Q^w
    # Mathematically: Q_c = T_u P_c T_v^T
    # Uses np.einsum for highly efficient 3D tensor multiplication
    Qw = np.einsum('ui, vj, ijc -> uvc', T_u, T_v, Pw)

    # 6. Project results back to 3D physical space
    W_new = Qw[..., 3]
    Q_new = np.empty((Qw.shape[0], Qw.shape[1], 3))
    Q_new[..., 0] = Qw[..., 0] / W_new
    Q_new[..., 1] = Qw[..., 1] / W_new
    Q_new[..., 2] = Qw[..., 2] / W_new

    refined = Surface_NURBS(
        control_points=Q_new,
        weights=W_new,
        knotvector_u=U_new,
        knotvector_v=V_new,
        degree_u=nurbs.degree_u,
        degree_v=nurbs.degree_v,
        ctrl_u=Q_new.shape[0],
        ctrl_v=Q_new.shape[1],
        torus_R=nurbs.torus_R,
        torus_r=nurbs.torus_r
    )

    return refined, T_u, T_v


# ============================================================================
# Knot insertion and degree elevation helpers (module-level, for p-refinement)
# ============================================================================

def _insert_single_knot_matrix_p(U: np.ndarray, p: int, u_bar: float):
    """Single knot insertion transfer matrix (for curves).

    Same logic as the internal function in refine_nurbs_h, but scoped independently.
    Returns (T, U_new), where T is an (n+1, n) matrix.
    """
    k = np.searchsorted(U, u_bar, side='right') - 1
    if k >= len(U) - p - 1:
        k = len(U) - p - 2

    n = len(U) - p - 1          # Original control point count
    T = np.zeros((n + 1, n))

    for i in range(n + 1):
        if i <= k - p:
            alpha = 1.0
        elif i >= k + 1:
            alpha = 0.0
        else:
            denominator = U[i + p] - U[i]
            alpha = (u_bar - U[i]) / denominator if denominator != 0 else 0.0

        if i < n:
            T[i, i] = alpha
        if i - 1 >= 0:
            T[i, i - 1] += (1.0 - alpha)

    U_new = np.insert(U, k + 1, u_bar)
    return T, U_new


def _insert_knots_curve(Pw: np.ndarray, U: np.ndarray, p: int,
                        knots: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Insert multiple knots into a single B-spline curve (homogeneous coords).

    Returns (Pw_new, U_new).
    """
    for u_bar in sorted(knots):
        T, U = _insert_single_knot_matrix_p(U, p, u_bar)
        Pw = T @ Pw
    return Pw, U


def _degree_elevate_curve(Pw: np.ndarray, U: np.ndarray,
                          p: int, t: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Degree-elevate a B-spline curve (homogeneous coords) by t orders.

    Returns (Pw_new, U_new, p_new) where p_new = p + t.
    """
    for _ in range(t):
        Pw, U = _degree_elevate_curve_one_step(Pw, U, p)
        p += 1
    return Pw, U, p


def _degree_elevate_curve_one_step(Pw: np.ndarray, U: np.ndarray, p: int):
    """Single degree-1 elevation implementation (Bezier decomposition + Bezier elevation)."""
    # Extract unique knots and multiplicities
    ub = [U[0]]
    for val in U[1:]:
        if val != ub[-1]:
            ub.append(val)
    ub = np.array(ub)
    mult = np.array([np.sum(U == v) for v in ub])

    # If internal knot multiplicity < p, insert knots to make piecewise Bezier
    insert_knots = []
    for i in range(1, len(ub) - 1):
        if mult[i] < p:
            insert_knots.extend([ub[i]] * (p - mult[i]))
    if insert_knots:
        Pw, U = _insert_knots_curve(Pw, U, p, insert_knots)

    # Curve is now piecewise Bezier (internal knot multiplicity = p)
    K = len(ub) - 1          # Bezier segment count
    pieces = []
    for i in range(K):
        start = i * p
        seg = Pw[start:start + p + 1]   # (p+1, 4)

        # Bezier elevation p -> p+1
        seg_new = np.zeros((p + 2, 4))
        seg_new[0] = seg[0]
        seg_new[p + 1] = seg[p]
        for j in range(1, p + 1):
            alpha = j / (p + 1)
            seg_new[j] = alpha * seg[j - 1] + (1 - alpha) * seg[j]

        if i == 0:
            pieces.append(seg_new)
        else:
            pieces.append(seg_new[1:])   # Share first control point

    Qw = np.vstack(pieces)              # (K*(p+1)+1, 4)

    # Construct elevated knot vector
    r = p + 1
    U_new = []
    U_new.extend([ub[0]] * (r + 1))
    for val in ub[1:-1]:
        U_new.extend([val] * r)
    U_new.extend([ub[-1]] * (r + 1))

    return Qw, np.array(U_new)


# ============================================================================
# Public API — p-refinement (degree elevation)
# ============================================================================

def _extract_degree_elevation_matrix(n, knotvector, degree, t):
    """Extract degree-elevation operator matrix T via column probing"""
    dummy = np.zeros((n, 4))
    dummy[:, 3] = 1.0   # <-- add this line
    dummy[0, 0] = 1.0
    result, U_new, _ = _degree_elevate_curve(dummy, knotvector.copy(), degree, t)
    new_n = result.shape[0]
    
    T = np.zeros((new_n, n))
    T[:, 0] = result[:, 0]
    
    for i in range(1, n):
        dummy = np.zeros((n, 4))
        dummy[i, 0] = 1.0
        result, _, _ = _degree_elevate_curve(dummy, knotvector.copy(), degree, t)
        T[:, i] = result[:, 0]
    
    return T, U_new


def refine_nurbs_p(nurbs: Surface_NURBS,
                   degree_u: int,
                   degree_v: int):
    """Perform p-refinement (degree elevation) on NURBS surface, preserving geometry.

    Returns
    -------
    refined : Surface_NURBS
    T_u : (new_ctrl_u, old_ctrl_u) ndarray
    T_v : (new_ctrl_v, old_ctrl_v) ndarray
    """
    if degree_u < nurbs.degree_u or degree_v < nurbs.degree_v:
        raise ValueError("Target degree must be >= current degree.")
    t_u = degree_u - nurbs.degree_u
    t_v = degree_v - nurbs.degree_v

    if t_u == 0 and t_v == 0:
        print("error")
        return nurbs, np.eye(nurbs.ctrl_u), np.eye(nurbs.ctrl_v)

    # ---- Extract T_u ----
    if t_u > 0:
        T_u, U_new = _extract_degree_elevation_matrix(
            nurbs.ctrl_u, nurbs.knotvector_u, nurbs.degree_u, t_u)
    else:
        T_u = np.eye(nurbs.ctrl_u)
        U_new = nurbs.knotvector_u.copy()

    # ---- Extract T_v ----
    if t_v > 0:
        T_v, V_new = _extract_degree_elevation_matrix(
            nurbs.ctrl_v, nurbs.knotvector_v, nurbs.degree_v, t_v)
    else:
        T_v = np.eye(nurbs.ctrl_v)
        V_new = nurbs.knotvector_v.copy()

    # ---- Compute new homogeneous control points via T_u, T_v ----
    Pw = np.empty((nurbs.ctrl_u, nurbs.ctrl_v, 4))
    W = nurbs.weights
    Pw[..., 0] = nurbs.control_points[..., 0] * W
    Pw[..., 1] = nurbs.control_points[..., 1] * W
    Pw[..., 2] = nurbs.control_points[..., 2] * W
    Pw[..., 3] = W

    Qw = np.einsum('ui, vj, ijc -> uvc', T_u, T_v, Pw)

    # Project back to 3D
    W_new = Qw[..., 3]
    Q_new = np.empty((Qw.shape[0], Qw.shape[1], 3))
    Q_new[..., 0] = Qw[..., 0] / W_new
    Q_new[..., 1] = Qw[..., 1] / W_new
    Q_new[..., 2] = Qw[..., 2] / W_new

    refined = Surface_NURBS(
        control_points=Q_new,
        weights=W_new,
        knotvector_u=U_new,
        knotvector_v=V_new,
        degree_u=degree_u,
        degree_v=degree_v,
        ctrl_u=Q_new.shape[0],
        ctrl_v=Q_new.shape[1],
        torus_R=nurbs.torus_R,
        torus_r=nurbs.torus_r,
    )

    return refined, T_u, T_v

def predict_extraction_p_refined_size(
        degree_u: int, degree_v: int, 
        ctrl_u: int, ctrl_v: int, 
        target_u_deg: int, target_v_deg: int,
        knotvector_u: np.ndarray = None, 
        knotvector_v: np.ndarray = None):
    """
    Estimate control point count for Bezier-extraction-based p-refinement (C^0 join algorithm).

    Parameters
    ----------
    degree_u, degree_v : int
        Current U and V degrees
    ctrl_u, ctrl_v : int
        Current U and V control point counts
    target_u_deg, target_v_deg : int
        Target U and V degrees
    knotvector_u, knotvector_v : ndarray, optional
        Optional. Passing knot vectors improves accuracy when many repeated knots exist.

    Returns
    -------
    new_ctrl_u : int
        p-refined U-direction control point count
    new_ctrl_v : int
        p-refined V-direction control point count
    """
    if target_u_deg < degree_u or target_v_deg < degree_v:
        raise ValueError("Target degree must be >= current degree.")

    # ---- 1. compute U-direction  Bezier segment count ----
    if knotvector_u is not None:
        internal_knots = knotvector_u[degree_u + 1 : len(knotvector_u) - (degree_u + 1)]
        segments_u = len(np.unique(internal_knots)) + 1
    else:
        # Default: assume internal knot multiplicity = 1 after regular h-refinement
        segments_u = ctrl_u - degree_u

    # ---- 2. compute V-direction  Bezier segment count ----
    if knotvector_v is not None:
        internal_knots = knotvector_v[degree_v + 1 : len(knotvector_v) - (degree_v + 1)]
        segments_v = len(np.unique(internal_knots)) + 1
    else:
        segments_v = ctrl_v - degree_v

    # ---- 3. Compute new control points via C^0 join logic ----
    # Formula: n_segments * target_degree + 1
    new_ctrl_u = segments_u * target_u_deg + 1
    new_ctrl_v = segments_v * target_v_deg + 1

    return new_ctrl_u, new_ctrl_v

from numpy.polynomial.legendre import leggauss
from typing import Dict, Any

def compute_iga_field_l2_error(
    nurbs1: Surface_NURBS, u_vec1: np.ndarray,
    nurbs2: Surface_NURBS, u_vec2: np.ndarray,
    n_gauss: int = 5
) -> float:
    """Compute true L2 error between two IGA physics fields using Gaussian quadrature.
    
    Partitions integration cells on [0,1]^2 based on knot vector union of both fields,
    accounting for geometric Jacobian (area element) of the surface.

    Parameters
    ----------
    nurbs1  : Surface_NURBS — First NURBS geometry (typically the reference domain)
    u_vec1  : ((ctrl_u1*ctrl_v1,) or (ctrl_u1, ctrl_v1) — control variable vector on nurbs1
    nurbs2  : Surface_NURBS — Second NURBS geometry
    u_vec2  : ((ctrl_u2*ctrl_v2,) or (ctrl_u2, ctrl_v2) — control variable vector on nurbs2
    n_gauss : int — Quadrature points per dimension (default 5, exact for polynomials up to degree 9)

    Returns
    -------
    L2_error : float — || u1 - u2 ||_{L2(\Omega)}
    """
    # 1. Reshape 1D control variable to 3D tensor (ctrl_u, ctrl_v, 1)
    # This allows direct reuse of _eval_nurbs_surface without rewriting interpolation
    U1_3d = u_vec1.reshape(nurbs1.ctrl_u, nurbs1.ctrl_v, 1)
    U2_3d = u_vec2.reshape(nurbs2.ctrl_u, nurbs2.ctrl_v, 1)

    # 2. Build common integration grid (Knot union)
    # Extract unique knots to form non-zero Bezier integration intervals
    u_knots = np.unique(np.concatenate([nurbs1.knotvector_u, nurbs2.knotvector_u]))
    v_knots = np.unique(np.concatenate([nurbs1.knotvector_v, nurbs2.knotvector_v]))
    
    # Ensure within valid parametric domain
    u_knots = u_knots[(u_knots >= 0.0) & (u_knots <= 1.0)]
    v_knots = v_knots[(v_knots >= 0.0) & (v_knots <= 1.0)]

    # 3. Get standard [-1,1] Gauss points and weights
    gp, gw = leggauss(n_gauss)
    gp_u, gp_v = np.meshgrid(gp, gp, indexing='ij')
    gw_u, gw_v = np.meshgrid(gw, gw, indexing='ij')
    
    gp_flat = np.stack([gp_u.ravel(), gp_v.ravel()], axis=-1)  # (n_gauss**2, 2)
    gw_flat = gw_u.ravel() * gw_v.ravel()                      # (n_gauss**2,)

    # 4. Vectorized mapping of all integration interval points
    global_uv = []
    global_weights = []

    for i in range(len(u_knots) - 1):
        for j in range(len(v_knots) - 1):
            ua, ub = u_knots[i], u_knots[i+1]
            va, vb = v_knots[j], v_knots[j+1]
            
            du, dv = ub - ua, vb - va
            if du < 1e-12 or dv < 1e-12:
                continue
                
            # Map from [-1,1] to actual parameter interval [ua, ub]
            mapped_u = ua + 0.5 * du * (gp_flat[:, 0] + 1.0)
            mapped_v = va + 0.5 * dv * (gp_flat[:, 1] + 1.0)
            
            # Local parameter mapping Jacobian: det(J_param) = (du/2)*(dv/2)
            detJ_param = (0.5 * du) * (0.5 * dv)
            
            global_uv.append(np.stack([mapped_u, mapped_v], axis=-1))
            global_weights.append(gw_flat * detJ_param)

    # Aggregate all Gauss points to evaluate (N_total, 2)
    global_uv = np.concatenate(global_uv, axis=0)
    global_weights = np.concatenate(global_weights, axis=0)

    # 5. Compute physical fields u1 and u2
    # Seamlessly reuse internal geometry interpolation
    val1 = _eval_nurbs_surface(
        global_uv, U1_3d, nurbs1.weights, 
        nurbs1.knotvector_u, nurbs1.knotvector_v, nurbs1.degree_u, nurbs1.degree_v
    ) # Returns shape (N, 1)
    
    val2 = _eval_nurbs_surface(
        global_uv, U2_3d, nurbs2.weights, 
        nurbs2.knotvector_u, nurbs2.knotvector_v, nurbs2.degree_u, nurbs2.degree_v
    ) # Returns shape (N, 1)

    # 6. Compute geometric Jacobian |J_geo| (area element) for physical space integration
    # Area element = | dS/du × dS/dv |
    # Use high-order central difference for fast partial derivatives, maintaining generality
    eps = 1e-6
    uv_u_plus = np.copy(global_uv); uv_u_plus[:, 0] = np.clip(uv_u_plus[:, 0] + eps, 0, 1)
    uv_u_minus = np.copy(global_uv); uv_u_minus[:, 0] = np.clip(uv_u_minus[:, 0] - eps, 0, 1)
    uv_v_plus = np.copy(global_uv); uv_v_plus[:, 1] = np.clip(uv_v_plus[:, 1] + eps, 0, 1)
    uv_v_minus = np.copy(global_uv); uv_v_minus[:, 1] = np.clip(uv_v_minus[:, 1] - eps, 0, 1)

    S_u_plus  = nurbs1.eval(uv_u_plus)
    S_u_minus = nurbs1.eval(uv_u_minus)
    S_v_plus  = nurbs1.eval(uv_v_plus)
    S_v_minus = nurbs1.eval(uv_v_minus)

    # Compute actual step size (Auto-degrade to forward/backward diff at boundaries)
    actual_du = uv_u_plus[:, 0] - uv_u_minus[:, 0]
    actual_dv = uv_v_plus[:, 1] - uv_v_minus[:, 1]

    dS_du = (S_u_plus - S_u_minus) / actual_du[:, None]
    dS_dv = (S_v_plus - S_v_minus) / actual_dv[:, None]

    normal_vec = np.cross(dS_du, dS_dv)
    J_geo = np.linalg.norm(normal_vec, axis=1)  # (N_total,)

    # 7. Aggregate integral: L2 = sqrt( sum (u1-u2)^2 * |J_geo| * detJ_param * weight )
    diff = val1[:, 0] - val2[:, 0]
    L2_error_sq = np.sum((diff ** 2) * J_geo * global_weights)

    return float(np.sqrt(L2_error_sq))


def compute_p2_nurbs_field_l2_error(
    P2: 'Surface_P2',
    u_vec1: np.ndarray,
    nurbs: 'Surface_NURBS',
    u_vec2: np.ndarray,
    n_gauss: int = 5
) -> float:
    """
    High-precision: compute L2 error in physical domain between P2 FEM and NURBS IGA solutions.
    
    Key improvements:
    1. Integration grid = NURBS unique knots ∪ P2 cell boundaries, eliminating cross-cell integration accuracy degradation
    2. 4th-order central difference for geometric Jacobian, reducing truncation error in high-curvature regions
    3. Scalar field interpolation strictly reuses P2 geometric shape functions for isoparametric consistency
    """

    # =========================================================================
    # 1. Build joint integration grid: NURBS knots ∪ P2 cell boundaries
    # =========================================================================
    u_knots_nurbs = np.unique(nurbs.knotvector_u)
    v_knots_nurbs = np.unique(nurbs.knotvector_v)

    # P2 uniform mesh boundaries
    u_breaks_p2 = np.linspace(0.0, 1.0, P2.n_u + 1)
    v_breaks_p2 = np.linspace(0.0, 1.0, P2.n_v + 1)

    # Merge and deduplicate (tol=1e-14 to avoid zero-length intervals from floating-point dupes)
    u_breaks = np.unique(np.concatenate([u_knots_nurbs, u_breaks_p2]))
    v_breaks = np.unique(np.concatenate([v_knots_nurbs, v_breaks_p2]))
    u_breaks = u_breaks[(u_breaks >= 0.0) & (u_breaks <= 1.0)]
    v_breaks = v_breaks[(v_breaks >= 0.0) & (v_breaks <= 1.0)]

    # Standard Gauss points and weights
    gp, gw = leggauss(n_gauss)
    gp_u, gp_v = np.meshgrid(gp, gp, indexing='ij')
    gw_u, gw_v = np.meshgrid(gw, gw, indexing='ij')
    gp_flat = np.stack([gp_u.ravel(), gp_v.ravel()], axis=-1)
    gw_flat = gw_u.ravel() * gw_v.ravel()

    # Map to all joint sub-intervals
    all_uv, all_w = [], []
    for i in range(len(u_breaks) - 1):
        for j in range(len(v_breaks) - 1):
            ua, ub = u_breaks[i], u_breaks[i + 1]
            va, vb = v_breaks[j], v_breaks[j + 1]
            du, dv = ub - ua, vb - va
            if du < 1e-14 or dv < 1e-14:
                continue
            mapped_u = ua + 0.5 * du * (gp_flat[:, 0] + 1.0)
            mapped_v = va + 0.5 * dv * (gp_flat[:, 1] + 1.0)
            detJ_param = (0.5 * du) * (0.5 * dv)
            all_uv.append(np.stack([mapped_u, mapped_v], axis=-1))
            all_w.append(gw_flat * detJ_param)

    global_uv = np.concatenate(all_uv, axis=0)
    global_weights = np.concatenate(all_w, axis=0)

    # =========================================================================
    # 2. Evaluate two physical fields
    # =========================================================================
    val_p2_scalar = _eval_p2_scalar_field(global_uv, P2, u_vec1)

    U2_3d = u_vec2.reshape(nurbs.ctrl_u, nurbs.ctrl_v, 1)
    val_nurbs = _eval_nurbs_surface(
        global_uv, U2_3d, nurbs.weights,
        nurbs.knotvector_u, nurbs.knotvector_v,
        nurbs.degree_u, nurbs.degree_v
    )[:, 0]

    # =========================================================================
    # 3. 4th-order central difference for geometric Jacobian |dS/du × dS/dv|
    # =========================================================================
    J_geo = _compute_surface_jacobian_high_order(nurbs, global_uv)

    # =========================================================================
    # 4. L2 error integration
    # =========================================================================
    diff = val_p2_scalar - val_nurbs
    L2_error_sq = np.sum((diff ** 2) * J_geo * global_weights)

    return float(np.sqrt(max(L2_error_sq, 0.0)))


def _compute_surface_jacobian_high_order(
    nurbs: 'Surface_NURBS', 
    uv: np.ndarray
) -> np.ndarray:
    """
    4th-order central difference for surface area element, O(h^4) accuracy.
    Compared to 2nd-order, significantly reduces truncation error in high-curvature regions.
    """
    eps = 1e-5

    def S(u, v):
        pts = np.stack([
            np.clip(u, 0, 1),
            np.clip(v, 0, 1)
        ], axis=-1)
        return nurbs.eval(pts)

    u0, v0 = uv[:, 0], uv[:, 1]

    # 4th-order central difference coefficients: f' ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    h_u = np.full_like(u0, eps)
    h_v = np.full_like(v0, eps)

    dS_du = (
        -S(u0 + 2*h_u, v0) + 8*S(u0 + h_u, v0)
        - 8*S(u0 - h_u, v0) + S(u0 - 2*h_u, v0)
    ) / (12.0 * eps)

    dS_dv = (
        -S(u0, v0 + 2*h_v) + 8*S(u0, v0 + h_v)
        - 8*S(u0, v0 - h_v) + S(u0, v0 - 2*h_v)
    ) / (12.0 * eps)

    normal = np.cross(dS_du, dS_dv)
    return np.linalg.norm(normal, axis=1)


def _eval_p2_scalar_field(
    uv: np.ndarray,
    p2: 'Surface_P2',
    u_vec: np.ndarray
) -> np.ndarray:
    """
    Evaluate P2 scalar field at arbitrary parameter points.
    Strictly reuses shape functions and node indexing logic from _eval_p2_at_arbitrary_points,
    only replaces 3D node coordinates with scalar DOFs for isoparametric mapping consistency.
    """
    n_u, n_v = p2.n_u, p2.n_v
    u_arr, v_arr = uv[:, 0], uv[:, 1]
    du, dv = 1.0 / n_u, 1.0 / n_v

    cell_i = np.clip((u_arr / du).astype(int), 0, n_u - 1)
    cell_j = np.clip((v_arr / dv).astype(int), 0, n_v - 1)

    xi = (u_arr - cell_i * du) / du
    eta = (v_arr - cell_j * dv) / dv

    in_tri1 = xi >= eta

    la = np.where(in_tri1, 1.0 - xi, 1.0 - eta)
    lb = np.where(in_tri1, xi - eta, xi)
    lc = np.where(in_tri1, eta, eta - xi)

    N1 = la * (2.0 * la - 1.0)
    N2 = lb * (2.0 * lb - 1.0)
    N3 = lc * (2.0 * lc - 1.0)
    N4 = 4.0 * la * lb
    N5 = 4.0 * lb * lc
    N6 = 4.0 * la * lc

    n_vert = (n_u + 1) * (n_v + 1)
    n_h = n_u * (n_v + 1)
    n_v_edge = (n_u + 1) * n_v

    idx_v00 = cell_j + cell_i * (n_v + 1)
    idx_v10 = cell_j + (cell_i + 1) * (n_v + 1)
    idx_v11 = (cell_j + 1) + (cell_i + 1) * (n_v + 1)
    idx_v01 = (cell_j + 1) + cell_i * (n_v + 1)

    idx_h00 = n_vert + cell_j + cell_i * (n_v + 1)
    idx_h01 = n_vert + (cell_j + 1) + cell_i * (n_v + 1)
    idx_v10_e = n_vert + n_h + cell_j + (cell_i + 1) * n_v
    idx_v00_e = n_vert + n_h + cell_j + cell_i * n_v
    idx_d00 = n_vert + n_h + n_v_edge + cell_j + cell_i * n_v

    idx1 = idx_v00
    idx2 = np.where(in_tri1, idx_v10, idx_v11)
    idx3 = np.where(in_tri1, idx_v11, idx_v01)
    idx4 = np.where(in_tri1, idx_h00, idx_d00)
    idx5 = np.where(in_tri1, idx_v10_e, idx_h01)
    idx6 = np.where(in_tri1, idx_d00, idx_v00_e)

    return (N1 * u_vec[idx1] + N2 * u_vec[idx2] + N3 * u_vec[idx3] +
            N4 * u_vec[idx4] + N5 * u_vec[idx5] + N6 * u_vec[idx6])


#--------precompute err---------

def precompute_iga_full(
    nurbs1: 'Surface_NURBS',
    u_vec1: np.ndarray,
    nurbs2: 'Surface_NURBS',
    n_gauss: int = 5
) -> Dict[str, Any]:
    """High-precision precompute: preserve repeated knot breakpoints + 4th-order geometric Jacobian."""

    # =================================================================
    # 1. Build integration grid: preserve repeated knots, don't blindly unique
    #    Only deduplicate within float tolerance to preserve C^0/C^1 breakpoints
    # =================================================================
    def _unique_knots_with_multiplicity(kv):
        """Preserve repeated knots, only merge truly identical floats within tol"""
        sorted_kv = np.sort(kv)
        mask = np.concatenate([[True], np.diff(sorted_kv) > 1e-14])
        return sorted_kv[mask]

    u_knots = _unique_knots_with_multiplicity(
        np.concatenate([nurbs1.knotvector_u, nurbs2.knotvector_u])
    )
    v_knots = _unique_knots_with_multiplicity(
        np.concatenate([nurbs1.knotvector_v, nurbs2.knotvector_v])
    )
    u_knots = u_knots[(u_knots >= 0.0) & (u_knots <= 1.0)]
    v_knots = v_knots[(v_knots >= 0.0) & (v_knots <= 1.0)]

    gp, gw = leggauss(n_gauss)
    gp_u, gp_v = np.meshgrid(gp, gp, indexing='ij')
    gw_u, gw_v = np.meshgrid(gw, gw, indexing='ij')
    gp_flat = np.stack([gp_u.ravel(), gp_v.ravel()], axis=-1)
    gw_flat = (gw_u * gw_v).ravel()

    all_uv, all_w = [], []
    for i in range(len(u_knots) - 1):
        for j in range(len(v_knots) - 1):
            ua, ub = u_knots[i], u_knots[i + 1]
            va, vb = v_knots[j], v_knots[j + 1]
            du, dv = ub - ua, vb - va
            if du < 1e-14 or dv < 1e-14:
                continue
            mapped_u = ua + 0.5 * du * (gp_flat[:, 0] + 1.0)
            mapped_v = va + 0.5 * dv * (gp_flat[:, 1] + 1.0)
            detJ_param = (0.5 * du) * (0.5 * dv)
            all_uv.append(np.stack([mapped_u, mapped_v], axis=-1))
            all_w.append(gw_flat * detJ_param)

    global_uv = np.concatenate(all_uv, axis=0)
    global_w = np.concatenate(all_w, axis=0)

    # =================================================================
    # 2. 4th-order central difference for geometric Jacobian(consistent with high-precision P2 version)
    # =================================================================
    J_geo = _compute_surface_jacobian_high_order(nurbs1, global_uv)

    # =================================================================
    # 3. Reference field u1 evaluation
    # =================================================================
    U1_3d = u_vec1.reshape(nurbs1.ctrl_u, nurbs1.ctrl_v, 1)
    u1_vals = _eval_nurbs_surface(
        global_uv, U1_3d, nurbs1.weights,
        nurbs1.knotvector_u, nurbs1.knotvector_v,
        nurbs1.degree_u, nurbs1.degree_v
    )[:, 0]

    return {
        'global_uv': global_uv,
        'global_w': global_w,
        'J_geo': J_geo,
        'u1_vals': u1_vals,
        'nurbs2': nurbs2,
    }


def compute_iga_l2_from_precomp(
    u_vec2: np.ndarray,
    precomp: Dict[str, Any]
) -> float:
    """Fast L2 error from precomputed data (logic unchanged, accuracy boost from precomputation)."""

    nurbs2 = precomp['nurbs2']
    U2_3d = u_vec2.reshape(nurbs2.ctrl_u, nurbs2.ctrl_v, 1)
    val2 = _eval_nurbs_surface(
        precomp['global_uv'], U2_3d, nurbs2.weights,
        nurbs2.knotvector_u, nurbs2.knotvector_v,
        nurbs2.degree_u, nurbs2.degree_v
    )[:, 0]

    diff = precomp['u1_vals'] - val2
    L2_sq = np.sum(diff * diff * precomp['J_geo'] * precomp['global_w'])
    return float(np.sqrt(max(L2_sq, 0.0)))


def precompute_l2_error_data(
    P2: 'Surface_P2',
    nurbs: 'Surface_NURBS',
    u_vec2: np.ndarray,
    n_gauss: int = 5
) -> Dict[str, Any]:
    """
    Precompute all quantities independent of u_vec1 for L2 error.
    
    When (P2, nurbs, u_vec2, n_gauss) are fixed and u_vec1 varies frequently,
    call this function once to avoid repeated integration grid, NURBS field evaluation, and geometric Jacobian computation.

    Parameters
    ----------
    P2 : Surface_P2
        P2 discretization object (used to determine P2 cell boundaries in joint integration grid)
    nurbs : Surface_NURBS
        NURBS surface object
    u_vec2 : np.ndarray
        NURBS control variable vector
    n_gauss : int
        Gauss quadrature points per parameter direction

    Returns
    -------
    dict
        Contains global_uv, global_weights, val_nurbs, J_geo, P2 ref, etc. precomputed data
    """
    # =========================================================================
    # 1. Build joint integration grid: NURBS knots ∪ P2 cell boundaries
    # =========================================================================
    u_knots_nurbs = np.unique(nurbs.knotvector_u)
    v_knots_nurbs = np.unique(nurbs.knotvector_v)
    u_breaks_p2 = np.linspace(0.0, 1.0, P2.n_u + 1)
    v_breaks_p2 = np.linspace(0.0, 1.0, P2.n_v + 1)

    u_breaks = np.unique(np.concatenate([u_knots_nurbs, u_breaks_p2]))
    v_breaks = np.unique(np.concatenate([v_knots_nurbs, v_breaks_p2]))
    u_breaks = u_breaks[(u_breaks >= 0.0) & (u_breaks <= 1.0)]
    v_breaks = v_breaks[(v_breaks >= 0.0) & (v_breaks <= 1.0)]

    gp, gw = leggauss(n_gauss)
    gp_u, gp_v = np.meshgrid(gp, gp, indexing='ij')
    gw_u, gw_v = np.meshgrid(gw, gw, indexing='ij')
    gp_flat = np.stack([gp_u.ravel(), gp_v.ravel()], axis=-1)
    gw_flat = gw_u.ravel() * gw_v.ravel()

    all_uv, all_w = [], []
    for i in range(len(u_breaks) - 1):
        for j in range(len(v_breaks) - 1):
            ua, ub = u_breaks[i], u_breaks[i + 1]
            va, vb = v_breaks[j], v_breaks[j + 1]
            du, dv = ub - ua, vb - va
            if du < 1e-14 or dv < 1e-14:
                continue
            mapped_u = ua + 0.5 * du * (gp_flat[:, 0] + 1.0)
            mapped_v = va + 0.5 * dv * (gp_flat[:, 1] + 1.0)
            detJ_param = (0.5 * du) * (0.5 * dv)
            all_uv.append(np.stack([mapped_u, mapped_v], axis=-1))
            all_w.append(gw_flat * detJ_param)

    global_uv = np.concatenate(all_uv, axis=0)
    global_weights = np.concatenate(all_w, axis=0)

    # =========================================================================
    # 2. Evaluate NURBS physics field (independent of u_vec1)
    # =========================================================================
    U2_3d = u_vec2.reshape(nurbs.ctrl_u, nurbs.ctrl_v, 1)
    val_nurbs = _eval_nurbs_surface(
        global_uv, U2_3d, nurbs.weights,
        nurbs.knotvector_u, nurbs.knotvector_v,
        nurbs.degree_u, nurbs.degree_v
    )[:, 0]

    # =========================================================================
    # 3. 4th-order central difference for geometric Jacobian(independent of u_vec1)
    # =========================================================================
    J_geo = _compute_surface_jacobian_high_order(nurbs, global_uv)

    return {
        "global_uv": global_uv,
        "global_weights": global_weights,
        "val_nurbs": val_nurbs,
        "J_geo": J_geo,
        "P2": P2,
    }


def compute_l2_error_from_precomputed(
    precomputed: Dict[str, Any],
    u_vec1: np.ndarray
) -> float:
    """
    Fast L2 error using precomputed data.
    
    Only needs P2 scalar field evaluation and weighted summation, avoiding integration grid,
    NURBS field evaluation, and geometric Jacobian recomputation overhead.

    Parameters
    ----------
    precomputed : dict
        Return value of precompute_l2_error_data
    u_vec1 : np.ndarray
        Nodal DOF vector on P2

    Returns
    -------
    float

        || u_P2 - u_NURBS ||_{L2(Omega)}
    """
    global_uv = precomputed["global_uv"]
    global_weights = precomputed["global_weights"]
    val_nurbs = precomputed["val_nurbs"]
    J_geo = precomputed["J_geo"]
    P2 = precomputed["P2"]

    # Only part needing recomputation: P2 scalar field evaluation
    val_p2_scalar = _eval_p2_scalar_field(global_uv, P2, u_vec1)

    diff = val_p2_scalar - val_nurbs
    L2_error_sq = np.sum((diff ** 2) * J_geo * global_weights)

    return float(np.sqrt(max(L2_error_sq, 0.0)))


#----------------err field------------


import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from matplotlib import cm

def compute_error_field_as_p2_surface(
    P2: 'Surface_P2',
    u_vec1: np.ndarray,
    nurbs: 'Surface_NURBS',
    u_vec2: np.ndarray,
    n_gauss: int = 5
) -> 'Surface_P2':
    """
    High-precision: compute P2 vs NURBS error field (L2 projection).
    Modification: strictly restrict output geometry to parametric domain [0,1]^2,
    X=u, Y=v, Z=error value.
    """
    # 1. Joint integration grid
    u_knots_nurbs = np.unique(nurbs.knotvector_u)
    v_knots_nurbs = np.unique(nurbs.knotvector_v)
    u_breaks_p2 = np.linspace(0.0, 1.0, P2.n_u + 1)
    v_breaks_p2 = np.linspace(0.0, 1.0, P2.n_v + 1)
    
    u_breaks = np.unique(np.concatenate([u_knots_nurbs, u_breaks_p2]))
    v_breaks = np.unique(np.concatenate([v_knots_nurbs, v_breaks_p2]))
    u_breaks = u_breaks[(u_breaks >= 0.0) & (u_breaks <= 1.0)]
    v_breaks = v_breaks[(v_breaks >= 0.0) & (v_breaks <= 1.0)]

    gp, gw = leggauss(n_gauss)
    gp_u, gp_v = np.meshgrid(gp, gp, indexing='ij')
    gw_u, gw_v = np.meshgrid(gw, gw, indexing='ij')
    gp_flat = np.stack([gp_u.ravel(), gp_v.ravel()], axis=-1)
    gw_flat = gw_u.ravel() * gw_v.ravel()

    all_uv, all_w = [], []
    for i in range(len(u_breaks) - 1):
        for j in range(len(v_breaks) - 1):
            ua, ub = u_breaks[i], u_breaks[i + 1]
            va, vb = v_breaks[j], v_breaks[j + 1]
            du, dv = ub - ua, vb - va
            if du < 1e-14 or dv < 1e-14: continue
            mapped_u = ua + 0.5 * du * (gp_flat[:, 0] + 1.0)
            mapped_v = va + 0.5 * dv * (gp_flat[:, 1] + 1.0)
            detJ_param = (0.5 * du) * (0.5 * dv)
            all_uv.append(np.stack([mapped_u, mapped_v], axis=-1))
            all_w.append(gw_flat * detJ_param)

    global_uv = np.concatenate(all_uv, axis=0)
    global_weights = np.concatenate(all_w, axis=0)

    # 2. Evaluate physics fields and compute error Diff
    val_p2_scalar = _eval_p2_scalar_field(global_uv, P2, u_vec1)
    U2_3d = u_vec2.reshape(nurbs.ctrl_u, nurbs.ctrl_v, 1)
    val_nurbs = _eval_nurbs_surface(
        global_uv, U2_3d, nurbs.weights,
        nurbs.knotvector_u, nurbs.knotvector_v,
        nurbs.degree_u, nurbs.degree_v
    )[:, 0]
    
    diff = val_p2_scalar - val_nurbs
    J_geo = _compute_surface_jacobian_high_order(nurbs, global_uv)

    # 3. Assemble M and F matrices
    n_nodes = P2.nodes_3d.shape[0]
    F = np.zeros(n_nodes)
    n_u, n_v = P2.n_u, P2.n_v
    u_arr, v_arr = global_uv[:, 0], global_uv[:, 1]
    du, dv = 1.0 / n_u, 1.0 / n_v

    cell_i = np.clip((u_arr / du).astype(int), 0, n_u - 1)
    cell_j = np.clip((v_arr / dv).astype(int), 0, n_v - 1)
    xi = (u_arr - cell_i * du) / du
    eta = (v_arr - cell_j * dv) / dv
    in_tri1 = xi >= eta

    la = np.where(in_tri1, 1.0 - xi, 1.0 - eta)
    lb = np.where(in_tri1, xi - eta, xi)
    lc = np.where(in_tri1, eta, eta - xi)
    N1 = la * (2.0 * la - 1.0); N2 = lb * (2.0 * lb - 1.0); N3 = lc * (2.0 * lc - 1.0)
    N4 = 4.0 * la * lb; N5 = 4.0 * lb * lc; N6 = 4.0 * la * lc

    n_vert = (n_u + 1) * (n_v + 1)
    n_h = n_u * (n_v + 1)
    n_v_edge = (n_u + 1) * n_v

    idx_v00 = cell_j + cell_i * (n_v + 1)
    idx_v10 = cell_j + (cell_i + 1) * (n_v + 1)
    idx_v11 = (cell_j + 1) + (cell_i + 1) * (n_v + 1)
    idx_v01 = (cell_j + 1) + cell_i * (n_v + 1)
    idx_h00 = n_vert + cell_j + cell_i * (n_v + 1)
    idx_h01 = n_vert + (cell_j + 1) + cell_i * (n_v + 1)
    idx_v10_e = n_vert + n_h + cell_j + (cell_i + 1) * n_v
    idx_v00_e = n_vert + n_h + cell_j + cell_i * n_v
    idx_d00 = n_vert + n_h + n_v_edge + cell_j + cell_i * n_v

    Ns = [N1, N2, N3, N4, N5, N6]
    idxs = [idx_v00, np.where(in_tri1, idx_v10, idx_v11), np.where(in_tri1, idx_v11, idx_v01),
            np.where(in_tri1, idx_h00, idx_d00), np.where(in_tri1, idx_v10_e, idx_h01),
            np.where(in_tri1, idx_d00, idx_v00_e)]

    row, col, data = [], [], []
    integrand_F = diff * J_geo * global_weights

    for k in range(6):
        np.add.at(F, idxs[k], Ns[k] * integrand_F)
        for l in range(6):
            row.append(idxs[k])
            col.append(idxs[l])
            data.append(Ns[k] * Ns[l] * J_geo * global_weights)

    M = sps.coo_matrix((np.concatenate(data), (np.concatenate(row), np.concatenate(col))), shape=(n_nodes, n_nodes)).tocsc()
    e_vec = spsolve(M, F)

    # 4. Force map to [0,1]^2 parametric domain
    nodes_3d_new = np.empty_like(P2.nodes_3d)
    nodes_3d_new[:, 0] = P2.nodes_2d[:, 0]  # X = u
    nodes_3d_new[:, 1] = P2.nodes_2d[:, 1]  # Y = v
    nodes_3d_new[:, 2] = e_vec              # Z = error

    return Surface_P2(
        nodes_2d=P2.nodes_2d.copy(), nodes_3d=nodes_3d_new,
        triangles=P2.triangles.copy(), n_u=P2.n_u, n_v=P2.n_v, h=P2.h
    )


def compute_error_field_as_nurbs_surface(
    nurbs1: 'Surface_NURBS',
    u_vec1: np.ndarray,
    nurbs2: 'Surface_NURBS',
    u_vec2: np.ndarray,
    n_gauss: int = 5
) -> 'Surface_NURBS':
    """
    High-precision: compute NURBS physics field error (L2 projection).
    Modification: restrict output control points to parametric domain [0,1]^2,
    Compute X,Y parametric coordinates via Greville abscissae, Z=error value.
    """
    def _unique_knots(kv):
        sorted_kv = np.sort(kv)
        return sorted_kv[np.concatenate([[True], np.diff(sorted_kv) > 1e-14])]

    u_knots = _unique_knots(np.concatenate([nurbs1.knotvector_u, nurbs2.knotvector_u]))
    v_knots = _unique_knots(np.concatenate([nurbs1.knotvector_v, nurbs2.knotvector_v]))
    u_knots = u_knots[(u_knots >= 0.0) & (u_knots <= 1.0)]
    v_knots = v_knots[(v_knots >= 0.0) & (v_knots <= 1.0)]

    gp, gw = leggauss(n_gauss)
    gp_u, gp_v = np.meshgrid(gp, gp, indexing='ij')
    gw_u, gw_v = np.meshgrid(gw, gw, indexing='ij')
    gp_flat = np.stack([gp_u.ravel(), gp_v.ravel()], axis=-1)
    gw_flat = (gw_u * gw_v).ravel()

    all_uv, all_w = [], []
    for i in range(len(u_knots) - 1):
        for j in range(len(v_knots) - 1):
            ua, ub = u_knots[i], u_knots[i + 1]
            va, vb = v_knots[j], v_knots[j + 1]
            du, dv = ub - ua, vb - va
            if du < 1e-14 or dv < 1e-14: continue
            mapped_u = ua + 0.5 * du * (gp_flat[:, 0] + 1.0)
            mapped_v = va + 0.5 * dv * (gp_flat[:, 1] + 1.0)
            all_uv.append(np.stack([mapped_u, mapped_v], axis=-1))
            all_w.append(gw_flat * (0.5 * du) * (0.5 * dv))

    global_uv = np.concatenate(all_uv, axis=0)
    global_weights = np.concatenate(all_w, axis=0)

    U1_3d = u_vec1.reshape(nurbs1.ctrl_u, nurbs1.ctrl_v, 1)
    u1_vals = _eval_nurbs_surface(global_uv, U1_3d, nurbs1.weights, nurbs1.knotvector_u, nurbs1.knotvector_v, nurbs1.degree_u, nurbs1.degree_v)[:, 0]

    U2_3d = u_vec2.reshape(nurbs2.ctrl_u, nurbs2.ctrl_v, 1)
    u2_vals = _eval_nurbs_surface(global_uv, U2_3d, nurbs2.weights, nurbs2.knotvector_u, nurbs2.knotvector_v, nurbs2.degree_u, nurbs2.degree_v)[:, 0]

    diff = u1_vals - u2_vals
    J_geo = _compute_surface_jacobian_high_order(nurbs1, global_uv)

    # L2 projection
    n_ctrl = nurbs1.ctrl_u * nurbs1.ctrl_v
    F = np.zeros(n_ctrl)
    u_arr, v_arr = global_uv[:, 0], global_uv[:, 1]
    
    spans_u = _find_spans(nurbs1.ctrl_u - 1, nurbs1.degree_u, u_arr, nurbs1.knotvector_u)
    spans_v = _find_spans(nurbs1.ctrl_v - 1, nurbs1.degree_v, v_arr, nurbs1.knotvector_v)
    Nu = _basis_funs(spans_u, u_arr, nurbs1.degree_u, nurbs1.knotvector_u)
    Nv = _basis_funs(spans_v, v_arr, nurbs1.degree_v, nurbs1.knotvector_v)

    p, q = nurbs1.degree_u, nurbs1.degree_v
    u_idx = spans_u[:, None] - p + np.arange(p + 1)[None, :]
    v_idx = spans_v[:, None] - q + np.arange(q + 1)[None, :]
    
    local_w = nurbs1.weights[u_idx[:, :, None], v_idx[:, None, :]]
    NuNv = np.einsum('na,nb->nab', Nu, Nv)
    R = (NuNv * local_w) / np.sum(NuNv * local_w, axis=(1, 2))[:, None, None]

    row, col, data = [], [], []
    integrand_F = diff * J_geo * global_weights

    for a in range(p + 1):
        for b in range(q + 1):
            idx_k = u_idx[:, a] * nurbs1.ctrl_v + v_idx[:, b]
            np.add.at(F, idx_k, R[:, a, b] * integrand_F)
            for c in range(p + 1):
                for d in range(q + 1):
                    idx_l = u_idx[:, c] * nurbs1.ctrl_v + v_idx[:, d]
                    row.append(idx_k); col.append(idx_l); data.append(R[:, a, b] * R[:, c, d] * J_geo * global_weights)

    M = sps.coo_matrix((np.concatenate(data), (np.concatenate(row), np.concatenate(col))), shape=(n_ctrl, n_ctrl)).tocsc()
    e_vec = spsolve(M, F)

    # Compute Greville abscissae to flatten control points in 2D u-v domain
    greville_u = np.array([np.sum(nurbs1.knotvector_u[i+1 : i+p+1])/p if p>0 else 0.5 for i in range(nurbs1.ctrl_u)])
    greville_v = np.array([np.sum(nurbs1.knotvector_v[j+1 : j+q+1])/q if q>0 else 0.5 for j in range(nurbs1.ctrl_v)])
    gu_grid, gv_grid = np.meshgrid(greville_u, greville_v, indexing='ij')

    Q_new = np.empty_like(nurbs1.control_points)
    Q_new[:, :, 0] = gu_grid        # X = Greville u
    Q_new[:, :, 1] = gv_grid        # Y = Greville v
    Q_new[:, :, 2] = e_vec.reshape(nurbs1.ctrl_u, nurbs1.ctrl_v)

    return Surface_NURBS(
        control_points=Q_new, weights=nurbs1.weights.copy(),
        knotvector_u=nurbs1.knotvector_u.copy(), knotvector_v=nurbs1.knotvector_v.copy(),
        degree_u=nurbs1.degree_u, degree_v=nurbs1.degree_v,
        ctrl_u=nurbs1.ctrl_u, ctrl_v=nurbs1.ctrl_v,
        torus_R=nurbs1.torus_R, torus_r=nurbs1.torus_r
    )


# ============================================================================
# New: dedicated 2D rendering function for error field visualization on [0,1]^2
# ============================================================================

def visualize_error_field_2d(
    error_model: 'Surface_P2 | Surface_NURBS',
    title: str = "Error Field in Parameter Domain [0,1]²",
    n_samples: int = 200,
    cmap: str = 'coolwarm',
    figsize: tuple = (10, 8),
    dpi: int = 120,
    show: bool = True
):
    """
    Receive error model, plot 2D square color heatmap.
    Use model eval mechanism, extract Z coordinate (error value) and map to color.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Uniformly samplingparametric domain [0,1]^2
    u_fine = np.linspace(0, 1, n_samples)
    v_fine = np.linspace(0, 1, n_samples)
    ug, vg = np.meshgrid(u_fine, v_fine, indexing='ij')
    uv_flat = np.stack([ug.ravel(), vg.ravel()], axis=-1)

    # Extract Z coordinate as error field scalar
    if isinstance(error_model, Surface_P2):
        X = _eval_p2_on_fine_grid(u_fine, v_fine, error_model.nodes_3d, error_model.n_u, error_model.n_v)
        Z_error = X[:, :, 2]
    elif isinstance(error_model, Surface_NURBS):
        X = error_model.eval(uv_flat).reshape(n_samples, n_samples, 3)
        Z_error = X[:, :, 2]
    else:
        raise TypeError("Only supports Surface_P2 or Surface_NURBS mapped via compute_error_field model")

    # Find symmetric max absolute error, ensuring coolwarm colormap zero centered
    max_abs_err = np.max(np.abs(Z_error))
    print(max_abs_err)
    if max_abs_err == 0: 
        max_abs_err = 1e-12
        
    max_abs_err = 0.9294873192388236

    norm = plt.Normalize(vmin=-max_abs_err, vmax=max_abs_err)
    
    # Render heatmap square
    im = ax.pcolormesh(ug, vg, Z_error, cmap=cmap, norm=norm, shading='gouraud')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label(r'Error $e(u,v)$', fontsize=12)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(r'Parameter $u$', fontsize=12)
    ax.set_ylabel(r'Parameter $v$', fontsize=12)
    
    # Force equal aspect ratio, render perfect [0,1]x[0,1] square
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    if show:
        plt.show()

    return fig

# ============================================================================
# Module export list
# ============================================================================




__all__ = [
    # Data structures
    'Surface_Truth',
    'Surface_NURBS',
    'Surface_P1',
    'Surface_P2',
    # Core functions
    'init_truth_model',
    'truth_to_nurbs',
    'load_nurbs_from_npz',
    'nurbs_to_p1',
    'nurbs_to_p2',
    'load_p1_from_npz',
    'load_p2_from_npz',
    # Visualization
    'visualize_models',
    'visualize_single_model',
    'visualize_trimmed_nurbs',
    # Pipeline
    'pipeline_truth_to_all',
    # Error computation
    'compute_geometry_error',
    'compute_geometry_error_nurbs',
    'compute_geometry_error_p1',
    'compute_geometry_error_p2',
    'compute_geometry_error',
    'compute_mapping_error_vs_nurbs'

    'refine_nurbs_h',
    'refine_nurbs_p',
    'predict_extraction_p_refined_size',

    'compute_iga_field_l2_error',
    'compute_p2_nurbs_field_l2_error',

    'precompute_iga_full',
    'compute_iga_l2_from_precomp',
    'precompute_l2_error_data',
    'compute_l2_error_from_precomputed',

    'compute_error_field_as_p2_surface',
    'compute_error_field_as_nurbs_surface',

    'visualize_error_field_2d'
]
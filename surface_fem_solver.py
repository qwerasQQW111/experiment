"""
surface_fem_solver.py — Unified surface FEM/IGA solver library
==================================================
Provides K, M matrix assembly and Galerkin solvers for three surface discretizations:

  1. Surface_NURBS  — IGA (isogeometric analysis, B-spline/NURBS basis)
  2. Surface_P1     — linear Lagrange triangles (Dziuk-Elliott covariant LB)
  3. Surface_P2     — quadratic Lagrange triangles (isoparametric, 7-point Dunavant quadrature)

Solvers (closed surfaces):
  - solve_poisson_zero_mean:  -Δu = f, zero-mean constraint (augmented Lagrange multiplier)

Solvers (open surfaces):
  - apply_dirichlet_bc:       apply Dirichlet boundary conditions
  - solve_poisson_dirichlet:  -Δu = f, Dirichlet BC

General solvers:
  - solve_helmholtz:          -Δu + c*u = f, with optional Dirichlet BC

Dependencies: numpy, scipy, surface_model (provides Surface_NURBS / Surface_P1 / Surface_P2 data classes)
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat

# ============================================================================
# Part 1: Common quadrature tools
# ============================================================================

def gauss_legendre_1d(n):
    """1D Gauss-Legendre quadrature nodes and weights."""
    from numpy.polynomial.legendre import leggauss
    return leggauss(n)


def make_quadrature_2d(x, w):
    """Tensor-product 2D quadrature rule."""
    Xi, Eta = np.meshgrid(x, x, indexing="ij")
    Wi, Wj = np.meshgrid(w, w, indexing="ij")
    return np.stack([Xi.ravel(), Eta.ravel()], axis=-1), (Wi * Wj).ravel()


# ---------------------------------------------------------------------------
# 7-point Dunavant quadrature (5th order, for P2 triangles)
# ---------------------------------------------------------------------------
SQRT15 = np.sqrt(15.0)
_A1 = (6.0 + SQRT15) / 21.0
_A2 = (6.0 - SQRT15) / 21.0
_W1 = (155.0 - SQRT15) / 2400.0
_W2 = (155.0 + SQRT15) / 2400.0

DUNAVANT_PTS = np.array([
    [1.0 / 3.0, 1.0 / 3.0],
    [_A2, _A2], [_A2, 1.0 - 2.0 * _A2], [1.0 - 2.0 * _A2, _A2],
    [_A1, _A1], [_A1, 1.0 - 2.0 * _A1], [1.0 - 2.0 * _A1, _A1],
], dtype=np.float64)

DUNAVANT_WTS = np.array([
    9.0 / 80.0,
    _W1, _W1, _W1,
    _W2, _W2, _W2,
], dtype=np.float64)


# ============================================================================
# Part 2: NURBS / B-spline basis function tools
# ============================================================================

def _basis_funs(span, u, p, U):
    """Cox-de Boor recurrence (nonzero basis functions)."""
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = N[r] / denom if denom != 0.0 else 0.0
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def _bspline_deriv(span, u, p, U):
    """B-spline basis function values and derivatives."""
    N = _basis_funs(span, u, p, U)
    if p == 0:
        return N, np.zeros_like(N)
    N1 = _basis_funs(span, u, p - 1, U)
    Nlow = np.zeros(p + 1)
    Nlow[1:] = N1
    dN = np.zeros(p + 1)
    for j in range(p + 1):
        i = span - p + j
        denom = U[i + p] - U[i]
        if denom != 0:
            dN[j] += p * Nlow[j] / denom
        if j < p:
            denom = U[i + p + 1] - U[i + 1]
            if denom != 0:
                dN[j] -= p * Nlow[j + 1] / denom
    return N, dN


def _nurbs_eval(u, v, span_u, span_v, nurbs):
    """NURBS shape function values, gradients, and local control point indices."""
    U, V = nurbs.knotvector_u, nurbs.knotvector_v
    p, q = nurbs.degree_u, nurbs.degree_v

    Nu, dNu = _bspline_deriv(span_u, u, p, U)
    Nv, dNv = _bspline_deriv(span_v, v, q, V)

    iu0, iv0 = span_u - p, span_v - q
    ids = np.array([(iu0 + i, iv0 + j) for i in range(p + 1) for j in range(q + 1)])

    w = nurbs.weights[iu0:iu0 + p + 1, iv0:iv0 + q + 1]
    R = np.outer(Nu, Nv)
    W = np.sum(w * R)
    Rvals = (w * R / W).ravel()

    dRdu_num = np.outer(dNu, Nv)
    dRdv_num = np.outer(Nu, dNv)
    dWdu = np.sum(w * dRdu_num)
    dWdv = np.sum(w * dRdv_num)
    dRdu = ((w * dRdu_num * W - w * R * dWdu) / (W * W)).ravel()
    dRdv = ((w * dRdv_num * W - w * R * dWdv) / (W * W)).ravel()

    return Rvals, dRdu, dRdv, ids


def _geometry(nurbs, active_ids, dRdu, dRdv):
    """Push-forward mapping Jacobian and metric tensor."""
    X = nurbs.control_points[active_ids[:, 0], active_ids[:, 1]]
    du = np.sum(X * dRdu[:, None], axis=0)
    dv = np.sum(X * dRdv[:, None], axis=0)
    G11 = du @ du
    G12 = du @ dv
    G22 = dv @ dv
    detG = G11 * G22 - G12 * G12
    G = np.array([[G11, G12], [G12, G22]])
    return du, dv, detG, G


# ============================================================================
# Part 3: K / M matrix assembly
# ============================================================================

# ---------------------------------------------------------------------------
# 3.1  NURBS assembly
# ---------------------------------------------------------------------------

def assemble_nurbs_matrices(nurbs, elems, nq=7):
    """Assemble NURBS FEM stiffness matrix K and mass matrix M.

    Parameters
    ----------
    nurbs : Surface_NURBS
        NURBS surface object (control points, knot vectors, degrees, weights).
    elems : list of tuple
        List of (span_u, span_v, ua, ub, va, vb).
    nq : int
        Number of 1D Gauss quadrature points.

    Returns
    -------
    K : (n_dof, n_dof) csr_matrix
    M : (n_dof, n_dof) csr_matrix
    """
    p, q = nurbs.degree_u, nurbs.degree_v
    n_dof = nurbs.ctrl_u * nurbs.ctrl_v

    K_lil = lil_matrix((n_dof, n_dof))
    M_lil = lil_matrix((n_dof, n_dof))

    xi_1d, w_1d = gauss_legendre_1d(nq)
    qp, qw = make_quadrature_2d(xi_1d, w_1d)

    for (su, sv, ua, ub, va, vb) in elems:
        Ke = np.zeros(((p + 1) * (q + 1),) * 2)
        Me = np.zeros_like(Ke)

        for k in range(len(qw)):
            xi_k, eta_k = qp[k]
            u = ua + 0.5 * (xi_k + 1) * (ub - ua)
            v = va + 0.5 * (eta_k + 1) * (vb - va)
            jac = 0.25 * (ub - ua) * (vb - va)

            R, dRu, dRv, ids = _nurbs_eval(u, v, su, sv, nurbs)
            _, _, detG, G = _geometry(nurbs, ids, dRu, dRv)
            if detG <= 1e-14:
                continue

            sqrtG = np.sqrt(detG)
            Ginv = np.linalg.inv(G)
            grad = np.column_stack([dRu, dRv])

            Ke += (grad @ Ginv @ grad.T) * sqrtG * jac * qw[k]
            Me += np.outer(R, R) * sqrtG * jac * qw[k]

        dof = ids[:, 0] * nurbs.ctrl_v + ids[:, 1]
        for a in range(len(dof)):
            for b in range(len(dof)):
                K_lil[dof[a], dof[b]] += Ke[a, b]
                M_lil[dof[a], dof[b]] += Me[a, b]

    return csr_matrix(K_lil), csr_matrix(M_lil)


def assemble_nurbs_load(nurbs, elems, f_interp, nq=7):
    """Assemble NURBS right-hand side load vector (L2 projection of source f onto surface).

    Parameters
    ----------
    nurbs : Surface_NURBS
    elems : list of tuple
    f_interp : callable
        Accepts (N, 2) parameter coordinate array, returns (N,) source values.
    nq : int

    Returns
    -------
    F : (n_dof,) ndarray
    """
    n_dof = nurbs.ctrl_u * nurbs.ctrl_v
    F = np.zeros(n_dof)

    xi_1d, w_1d = gauss_legendre_1d(nq)
    qp, qw = make_quadrature_2d(xi_1d, w_1d)

    for (su, sv, ua, ub, va, vb) in elems:
        for k in range(len(qw)):
            xi_k, eta_k = qp[k]
            u = ua + 0.5 * (xi_k + 1) * (ub - ua)
            v = va + 0.5 * (eta_k + 1) * (vb - va)
            jac = 0.25 * (ub - ua) * (vb - va)

            R, dRu, dRv, ids = _nurbs_eval(u, v, su, sv, nurbs)
            _, _, detG, _ = _geometry(nurbs, ids, dRu, dRv)
            if detG <= 1e-14:
                continue

            sqrtG = np.sqrt(detG)
            f_val = float(f_interp(np.array([[u, v]]))[0])
            F_elem = R * f_val * sqrtG * jac * qw[k]
            dof = ids[:, 0] * nurbs.ctrl_v + ids[:, 1]
            np.add.at(F, dof, F_elem)

    return F


# ---------------------------------------------------------------------------
# 3.2  P1 assembly (Dziuk-Elliott covariant Laplace-Beltrami)
# ---------------------------------------------------------------------------

def assemble_p1_matrices(p1):
    """Assemble P1 linear Lagrange triangle stiffness K and mass M matrices.

    Based on Dziuk-Elliott covariant Laplace-Beltrami operator: DF^T DF consistent push-forward.

    Parameters
    ----------
    p1 : Surface_P1
        Contains nodes_2d, nodes_3d, triangles attributes.

    Returns
    -------
    K : (n_nodes, n_nodes) csr_matrix
    M : (n_nodes, n_nodes) csr_matrix
    """
    n_nodes = p1.nodes_2d.shape[0]
    n_elems = p1.triangles.shape[0]
    nodes_3d = p1.nodes_3d
    triangles = p1.triangles

    K_lil = lil_matrix((n_nodes, n_nodes))
    M_lil = lil_matrix((n_nodes, n_nodes))

    # Reference element gradients: grad(lambda_1), grad(lambda_2), grad(lambda_3) = -(grad(lambda_1)+grad(lambda_2))
    grad_ref = np.array([[-1.0, -1.0],
                         [1.0, 0.0],
                         [0.0, 1.0]])

    for e in range(n_elems):
        idx = triangles[e]
        X = nodes_3d[idx]                              # (3, 3)

        DF = np.column_stack([X[1] - X[0], X[2] - X[0]])  # (3, 2)
        G = DF.T @ DF
        detG = np.linalg.det(G)
        if detG < 1e-30:
            continue
        sqrt_detG = np.sqrt(detG)
        G_inv = np.linalg.inv(G)

        # Stiffness: K_e(i,j) = (grad_lam_i)^T G^{-1} (grad_lam_j) * sqrt(detG) * |T_ref|
        Ke = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Ke[i, j] = (grad_ref[i] @ G_inv @ grad_ref[j]) * sqrt_detG * 0.5

        # Mass: reference element M_ref
        Me_ref = np.full((3, 3), 0.5 / 12.0)
        np.fill_diagonal(Me_ref, 0.5 / 6.0)
        Me = sqrt_detG * Me_ref

        for i in range(3):
            for j in range(3):
                K_lil[idx[i], idx[j]] += Ke[i, j]
                M_lil[idx[i], idx[j]] += Me[i, j]

    return csr_matrix(K_lil), csr_matrix(M_lil)


# ---------------------------------------------------------------------------
# 3.3  P2 assembly (isoparametric quadratic, 7-point Dunavant quadrature)
# ---------------------------------------------------------------------------

def _p2_shape_functions(lam1, lam2):
    """P2 shape functions (6 nodes, reference triangle)."""
    lam3 = 1.0 - lam1 - lam2
    N = np.empty(6)
    N[0] = lam3 * (2.0 * lam3 - 1.0)
    N[1] = lam1 * (2.0 * lam1 - 1.0)
    N[2] = lam2 * (2.0 * lam2 - 1.0)
    N[3] = 4.0 * lam3 * lam1
    N[4] = 4.0 * lam1 * lam2
    N[5] = 4.0 * lam3 * lam2
    return N


def _p2_shape_gradients(lam1, lam2):
    """P2 shape function gradients (dN_a/dlam_1, dN_a/dlam_2)."""
    lam3 = 1.0 - lam1 - lam2
    dN = np.empty((6, 2))
    dN[0, 0] = dN[0, 1] = 1.0 - 4.0 * lam3
    dN[1, 0] = 4.0 * lam1 - 1.0
    dN[1, 1] = 0.0
    dN[2, 0] = 0.0
    dN[2, 1] = 4.0 * lam2 - 1.0
    dN[3, 0] = 4.0 * (lam3 - lam1)
    dN[3, 1] = -4.0 * lam1
    dN[4, 0] = 4.0 * lam2
    dN[4, 1] = 4.0 * lam1
    dN[5, 0] = -4.0 * lam2
    dN[5, 1] = 4.0 * (lam3 - lam2)
    return dN


def assemble_p2_matrices(p2, quadrature_pts=None, quadrature_wts=None):
    """Assemble P2 quadratic Lagrange triangle stiffness K and mass M matrices.

    Dziuk-Elliott covariant Laplace-Beltrami, isoparametric (geometry and solution share same P2 shape functions).
    Default: 7-point Dunavant Gaussian quadrature (5th order accuracy).

    Parameters
    ----------
    p2 : Surface_P2
    quadrature_pts : (N_qp, 2) ndarray, optional
    quadrature_wts : (N_qp,) ndarray, optional

    Returns
    -------
    K : (n_nodes, n_nodes) csr_matrix
    M : (n_nodes, n_nodes) csr_matrix
    """
    if quadrature_pts is None:
        quadrature_pts = DUNAVANT_PTS
    if quadrature_wts is None:
        quadrature_wts = DUNAVANT_WTS

    nodes_3d = p2.nodes_3d
    triangles = p2.triangles
    n_nodes = nodes_3d.shape[0]
    n_elems = triangles.shape[0]
    n_qp = len(quadrature_wts)

    K_lil = lil_matrix((n_nodes, n_nodes))
    M_lil = lil_matrix((n_nodes, n_nodes))

    for e in range(n_elems):
        idx = triangles[e]
        X_local = nodes_3d[idx]

        Ke = np.zeros((6, 6))
        Me = np.zeros((6, 6))

        for q in range(n_qp):
            lam1, lam2 = quadrature_pts[q]
            w = quadrature_wts[q]

            N = _p2_shape_functions(lam1, lam2)
            dN = _p2_shape_gradients(lam1, lam2)

            DF = X_local.T @ dN   # (3, 6) @ (6, 2) = (3, 2)
            G = DF.T @ DF
            detG = np.linalg.det(G)
            if detG < 1e-30:
                continue
            sqrt_detG = np.sqrt(detG)
            G_inv = np.linalg.inv(G)

            Ke += (dN @ G_inv @ dN.T) * (w * sqrt_detG)
            Me += np.outer(N, N) * (w * sqrt_detG)

        for i in range(6):
            for j in range(6):
                K_lil[idx[i], idx[j]] += Ke[i, j]
                M_lil[idx[i], idx[j]] += Me[i, j]

    return csr_matrix(K_lil), csr_matrix(M_lil)


# ============================================================================
# Part 4: Unified Galerkin solvers
# ============================================================================

def solve_poisson_zero_mean(K, M, F):
    """Solve Poisson equation -Δu = f on closed surface, with constraint int_Γ u dΓ = 0.

    Uses augmented Lagrange multiplier to handle K's nullspace (constant function):
        [ K       M·1 ] [ u ]   [ F ]
        [ (M·1)^T  0  ] [ λ ] = [ 0 ]

    Parameters
    ----------
    K : (n, n) sparse matrix
        Stiffness matrix (singular, rank n-1).
    M : (n, n) sparse matrix
        Mass matrix.
    F : (n,) ndarray
        Right-hand side load vector.

    Returns
    -------
    u : (n,) ndarray
        Zero-mean solution.
    lambda_lm : float
        Lagrange multiplier.
    """
    n = K.shape[0]
    ones_n = np.ones(n)
    M_ones = M @ ones_n

    K_aug = bmat([[K,                        M_ones.reshape(-1, 1)],
                   [M_ones.reshape(1, -1),   np.zeros((1, 1))]],
                  format='csr')
    rhs_aug = np.concatenate([F, [0.0]])
    sol_aug = spsolve(K_aug, rhs_aug)

    return sol_aug[:-1], sol_aug[-1]


def apply_dirichlet_bc(A, F, bc_dofs, bc_vals=None):
    """Apply Dirichlet boundary conditions to sparse A and rhs F (row-column elimination).

    Parameters
    ----------
    A : (n, n) sparse matrix
        Stiffness matrix or system matrix.
    F : (n,) ndarray
        Right-hand side load vector.
    bc_dofs : (n_bc,) array_like
        DOF indices to apply BC on.
    bc_vals : (n_bc,) array_like, optional
        Values at boundary DOFs. If None, defaults to homogeneous BC (0).

    Returns
    -------
    A_bc : (n, n) csr_matrix
        Modified matrix.
    F_bc : (n,) ndarray
        Modified RHS vector.
    """
    A = A.tolil()
    F = F.copy()
    if bc_vals is None:
        bc_vals = np.zeros(len(bc_dofs))
    else:
        bc_vals = np.asarray(bc_vals)

    # 1. Adjust RHS F, subtract boundary column contribution to interior nodes
    F -= A[:, bc_dofs].dot(bc_vals)

    # 2. Zero out rows and columns of boundary DOFs, set diagonal to 1
    for dof, val in zip(bc_dofs, bc_vals):
        A[dof, :] = 0.0
        A[:, dof] = 0.0
        A[dof, dof] = 1.0
        F[dof] = val

    return A.tocsr(), F


def solve_poisson_dirichlet(K, F, bc_dofs, bc_vals=None):
    """Solve Poisson equation -Δu = f on open surface with Dirichlet BC.

    Parameters
    ----------
    K : (n, n) sparse matrix
        Stiffness matrix (singular).
    F : (n,) ndarray
        Right-hand side load vector.
    bc_dofs : (n_bc,) array_like
        Boundary DOF indices.
    bc_vals : (n_bc,) array_like, optional
        Boundary values. Default: homogeneous BC (0).

    Returns
    -------
    u : (n,) ndarray
        Numerical solution.
    """
    if bc_dofs is None or len(bc_dofs) == 0:
        raise ValueError("Single-patch open surface Poisson solve requires Dirichlet boundary DOFs bc_dofs.")

    A_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_vals)
    return spsolve(A_bc, F_bc)


def solve_helmholtz(K, M, c, F, bc_dofs=None, bc_vals=None):
    """Solve Helmholtz equation -Δu + c*u = f, with optional Dirichlet BC.

    Closed surface (bc_dofs=None): directly solve (K + c*M) u = F.
    Open surface (bc_dofs non-empty): apply Dirichlet BC then solve.

    Parameters
    ----------
    K : (n, n) sparse matrix
    M : (n, n) sparse matrix
    c : float
        Reaction coefficient.
    F : (n,) ndarray
    bc_dofs : (n_bc,) array_like, optional
        Boundary DOF indices.Pass None for closed surfaces; required for open surfaces.
    bc_vals : (n_bc,) array_like, optional
        Boundary values. Default: homogeneous BC (0).

    Returns
    -------
    u : (n,) ndarray
    """
    A = K + c * M
    if bc_dofs is not None and len(bc_dofs) > 0:
        A, F = apply_dirichlet_bc(A, F, bc_dofs, bc_vals)
    return spsolve(A, F)


# ============================================================================
# Part 5: Post-processing — L2 error
# ============================================================================

def l2_error_nurbs(nurbs, elems, u_h, u_true_interp, nq=7):
    """L2 error for NURBS solution (surface quadrature).

    Parameters
    ----------
    nurbs : Surface_NURBS
    elems : list of tuple
    u_h : (n_dof,) ndarray — numerical solution coefficients
    u_true_interp : callable — true solution interpolator
    nq : int

    Returns
    -------
    l2_err : float
    l2_norm_true : float
    """
    xi_1d, w_1d = gauss_legendre_1d(nq)
    qp, qw = make_quadrature_2d(xi_1d, w_1d)

    err_sq = 0.0
    true_sq = 0.0

    for (su, sv, ua, ub, va, vb) in elems:
        for k in range(len(qw)):
            xi_k, eta_k = qp[k]
            upar = ua + 0.5 * (xi_k + 1) * (ub - ua)
            vpar = va + 0.5 * (eta_k + 1) * (vb - va)
            jac = 0.25 * (ub - ua) * (vb - va)

            R, dRu, dRv, ids = _nurbs_eval(upar, vpar, su, sv, nurbs)
            _, _, detG, _ = _geometry(nurbs, ids, dRu, dRv)
            if detG <= 1e-14:
                continue

            dS = np.sqrt(detG) * jac * qw[k]
            dof = ids[:, 0] * nurbs.ctrl_v + ids[:, 1]
            uh_q = R @ u_h[dof]
            ut_q = float(u_true_interp(np.array([[upar, vpar]]))[0])

            err_sq += (uh_q - ut_q) ** 2 * dS
            true_sq += ut_q ** 2 * dS

    return np.sqrt(err_sq), np.sqrt(true_sq)


def l2_error(u_h, u_ref, M):
    """L2 error (discrete form, using mass matrix).

    err = √[(u_h - u_ref)^T M (u_h - u_ref)]
    norm_ref = √[u_ref^T M u_ref]

    Parameters
    ----------
    u_h : (n,) ndarray — numerical solution
    u_ref : (n,) ndarray — reference solution (on same nodes)
    M : (n, n) sparse matrix — mass matrix

    Returns
    -------
    l2_err : float
    l2_norm_ref : float
    """
    diff = u_h - u_ref
    err = np.sqrt(diff @ (M @ diff))
    norm = np.sqrt(u_ref @ (M @ u_ref))
    return err, norm

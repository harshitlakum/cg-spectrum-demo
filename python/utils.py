# utils.py
import numpy as np
import scipy.sparse as sp

def K1d(n):
    """1D Laplacian tridiagonal matrix (scipy sparse format)"""
    return sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')

def apply_A(x, K, n):
    """Efficiently apply A = I⊗K + K⊗I to vector x, with K (n x n) and x (N,)"""
    X = x.reshape((n, n))
    return (K @ X + X @ K.T).reshape(-1)

def energy_norm(e, K, n):
    """Compute sqrt(e^T A e) where A = I⊗K + K⊗I"""
    Ae = apply_A(e, K, n)
    return np.sqrt(e @ Ae)

def cg_method(Afunc, b, x0, K, n, x_star=None, maxit=200, tol=1e-8):
    x = x0.copy()
    r = b - Afunc(x)
    p = r.copy()
    errors = []
    if x_star is not None:
        e0 = x_star - x0
        norm0 = energy_norm(e0, K, n)
    else:
        norm0 = None
    for _ in range(maxit):
        Ap = Afunc(p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        if x_star is not None:
            e = x_star - x_new
            relerr = energy_norm(e, K, n) / norm0
            errors.append(relerr)
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        x = x_new
        r = r_new
    return x, np.array(errors) if errors else None

def steepest_descent(Afunc, b, x0, K, n, x_star=None, maxit=200, tol=1e-8):
    x = x0.copy()
    errors = []
    if x_star is not None:
        e0 = x_star - x0
        norm0 = energy_norm(e0, K, n)
    else:
        norm0 = None
    for _ in range(maxit):
        r = b - Afunc(x)
        Ar = Afunc(r)
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x_new = x + alpha * r
        if x_star is not None:
            relerr = energy_norm(x_star - x_new, K, n) / norm0
            errors.append(relerr)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, np.array(errors) if errors else None

def cg_spectrum_demo(A, b, x_star, x0, maxiter=30):
    """CG with A-norm error tracking for arbitrary SPD A"""
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    errors = []
    e0 = x_star - x0
    norm0 = np.sqrt(e0 @ A @ e0)
    for _ in range(maxiter):
        Ap = A @ p
        alpha = r @ r / (p @ Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        e = x_star - x_new
        relerr = np.sqrt(e @ A @ e) / norm0
        errors.append(relerr)
        if np.linalg.norm(r_new) < 1e-12:
            break
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        x = x_new
        r = r_new
    return np.array(errors)

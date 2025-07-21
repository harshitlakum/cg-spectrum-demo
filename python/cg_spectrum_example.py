# cg_spectrum_example.py
import numpy as np
import matplotlib.pyplot as plt
from utils import cg_spectrum_demo

np.random.seed(0)
n = 30
m = 3

# Create eigenvalues: m large, n-m nearly 1
eigvals = np.ones(n)
eigvals[-m:] = [100, 200, 300]
eigvals[:n-m] += np.linspace(0, 1e-3, n-m)

# Random orthogonal Q
Q, _ = np.linalg.qr(np.random.randn(n, n))
A = Q @ np.diag(eigvals) @ Q.T

# Exact solution and right-hand side
x_star = np.random.randn(n)
b = A @ x_star
x0 = np.zeros(n)

# Run CG
errors = cg_spectrum_demo(A, b, x_star, x0, maxiter=n)

# Plot
plt.figure(figsize=(7,5))
plt.semilogy(errors, 'bo-', label='CG relative A-norm error')
plt.axvline(m, color='r', linestyle='--', label=f'm = {m} (num large eigs)')
plt.xlabel('Iteration')
plt.ylabel(r'Relative energy error $\|x^*-x_k\|_A / \|x^*-x_0\|_A$')
plt.title('CG Convergence: $m$ large eigenvalues, rest clustered at 1')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

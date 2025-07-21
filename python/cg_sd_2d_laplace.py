# cg_sd_2d_laplace.py
import numpy as np
import matplotlib.pyplot as plt
from utils import K1d, apply_A, cg_method, steepest_descent

def theoretical_cg_bound(kappa, kmax):
    k = np.arange(1, kmax+1)
    return 2 * ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k

Ns = [100, 10000]  # Try 1_000_000 if you like
maxit = 50
plt.figure(figsize=(10,6))

for N in Ns:
    n = int(np.sqrt(N))
    print(f"Solving for N={N}, n={n} ...")
    K = K1d(n)
    x_star = np.random.rand(N)
    Afunc = lambda x: apply_A(x, K, n)
    b = Afunc(x_star)
    x0 = np.zeros(N)

    _, err_sd = steepest_descent(Afunc, b, x0, K, n, x_star, maxit=maxit)
    _, err_cg = cg_method(Afunc, b, x0, K, n, x_star, maxit=maxit)

    # Theoretical CG bound
    lambda_min = 4*(np.sin(np.pi/(2*(n+1)))**2)
    lambda_max = 4*(np.sin(n*np.pi/(2*(n+1)))**2)
    kappa = lambda_max / lambda_min
    th_bound = theoretical_cg_bound(kappa, len(err_cg))

    plt.semilogy(range(1,1+len(err_sd)), err_sd, 'o-', label=f'SD N={N}')
    plt.semilogy(range(1,1+len(err_cg)), err_cg, 's-', label=f'CG N={N}')
    plt.semilogy(range(1,1+len(th_bound)), th_bound, '--', label=f'CG Bound N={N}')

plt.xlabel('Iteration')
plt.ylabel(r'Relative energy error: $\|e_k\|_A/\|e_0\|_A$')
plt.title('SD vs CG Convergence on 2D Laplace System')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

function [x, relerr] = cg(Afunc, b, x0, K, n, x_star, maxit, tol)
    x = x0;
    r = b - Afunc(x);
    p = r;
    e0 = x0 - x_star;
    norm0 = energy_norm(e0, K, n);
    relerr = zeros(maxit, 1);
    for k = 1:maxit
        Ap = Afunc(p);
        alpha = (r'*r) / (p'*Ap);
        x_new = x + alpha * p;
        r_new = r - alpha * Ap;
        err = energy_norm(x_new - x_star, K, n) / norm0;
        relerr(k) = err;
        if err < tol
            relerr = relerr(1:k);
            break
        end
        beta = (r_new'*r_new) / (r'*r);
        p = r_new + beta * p;
        x = x_new; r = r_new;
    end
end
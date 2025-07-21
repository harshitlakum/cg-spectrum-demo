function [x, relerr] = sd(Afunc, b, x0, K, n, x_star, maxit, tol)
    x = x0;
    e0 = x0 - x_star;
    norm0 = energy_norm(e0, K, n);
    relerr = zeros(maxit, 1);
    for k = 1:maxit
        r = b - Afunc(x);
        Ar = Afunc(r);
        alpha = (r'*r) / (r'*Ar);
        x_new = x + alpha * r;
        err = energy_norm(x_new - x_star, K, n) / norm0;
        relerr(k) = err;
        if err < tol
            relerr = relerr(1:k);
            break
        end
        x = x_new;
    end
end

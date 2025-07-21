% cg_sd_2d_laplace.m
% SD vs CG for 2D Laplace, using Kronecker structure

Ns = [100, 10000];
maxit = 50; tol = 1e-8;
figure; hold on;

for idx = 1:length(Ns)
    N = Ns(idx);
    n = round(sqrt(N));
    K = K1d(n);
    x_star = rand(N,1);
    Afunc = @(x) apply_A(x, K, n);
    b = Afunc(x_star);
    x0 = zeros(N,1);

    [~, err_sd] = sd(Afunc, b, x0, K, n, x_star, maxit, tol);
    [~, err_cg] = cg(Afunc, b, x0, K, n, x_star, maxit, tol);

    % Theoretical CG bound
    lambda_min = 4*(sin(pi/(2*(n+1)))^2);
    lambda_max = 4*(sin(n*pi/(2*(n+1)))^2);
    kappa = lambda_max / lambda_min;
    bound = cg_bound(kappa, length(err_cg));

    semilogy(1:length(err_sd), err_sd, 'o-', 'DisplayName', sprintf('SD N=%d',N));
    semilogy(1:length(err_cg), err_cg, 's-', 'DisplayName', sprintf('CG N=%d',N));
    semilogy(1:length(bound), bound, '--', 'DisplayName', sprintf('CG Bound N=%d',N));
end

xlabel('Iteration');
ylabel('Relative energy norm error');
legend show;
title('SD vs CG Convergence for 2D Laplace System');
grid on;

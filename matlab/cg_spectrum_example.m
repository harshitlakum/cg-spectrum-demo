% cg_spectrum_example.m
% Illustrates CG convergence for a matrix with m large eigenvalues and the rest clustered near 1.

rng(0);
n = 30;
m = 3; % Number of large eigenvalues

eigvals = ones(n,1);
eigvals(end-m+1:end) = [100; 200; 300];
eigvals(1:n-m) = eigvals(1:n-m) + linspace(0,1e-3,n-m)'; % Small spread

[Q,~] = qr(randn(n,n));
A = Q*diag(eigvals)*Q';

x_star = randn(n,1);
b = A*x_star;
x0 = zeros(n,1);

maxiter = n;
errs = zeros(maxiter,1);
x = x0;
r = b - A*x;
p = r;
e0 = x_star - x0;
norm0 = sqrt(e0'*A*e0);

for k = 1:maxiter
    Ap = A*p;
    alpha = (r'*r)/(p'*Ap);
    x_new = x + alpha*p;
    r_new = r - alpha*Ap;
    e = x_star - x_new;
    errs(k) = sqrt(e'*A*e)/norm0;
    if norm(r_new) < 1e-12
        errs = errs(1:k);
        break;
    end
    beta = (r_new'*r_new)/(r'*r);
    p = r_new + beta*p;
    x = x_new;
    r = r_new;
end

figure;
semilogy(1:length(errs), errs, 'bo-','LineWidth',1.5); hold on
xline(m, 'r--','LineWidth',1.5,'DisplayName',sprintf('m = %d',m));
xlabel('Iteration');
ylabel('Relative energy error ||x^*-x_k||_A / ||x^*-x_0||_A');
title('CG Convergence: m large eigenvalues, rest clustered at 1');
legend('CG relative A-norm error',sprintf('m = %d',m),'Location','southwest');
grid on;

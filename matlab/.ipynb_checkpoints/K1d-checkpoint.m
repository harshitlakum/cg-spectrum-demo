function K = K1d(n)
% 1D Laplacian tridiagonal matrix (sparse)
    e = ones(n,1);
    K = spdiags([-e 2*e -e], -1:1, n, n);
end

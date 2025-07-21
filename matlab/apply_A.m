function y = apply_A(x, K, n)
% Efficiently apply A = kron(I,K)+kron(K,I) to vector x
    X = reshape(x, n, n);
    Y = K*X + X*K';
    y = Y(:);
end

function normA = energy_norm(e, K, n)
% Compute the A-norm sqrt(e' A e) where A = I⊗K + K⊗I
    Ae = apply_A(e, K, n);
    normA = sqrt(e'*Ae);
end

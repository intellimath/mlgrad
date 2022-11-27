import numpy as np
from numpy import diag, einsum
from numpy.linalg import det, inv, pinv

def scatter_matrix(X):
    return X.T @ X / len(X)

def robust_scatter_matrix(X, maf, tol=1.0e-8, n_iter=100):
    N = len(X)
    S = X.T @ X
    n1 = 1.0 / S.shape[0]
    S = pinv(S)
    S /= det(S) ** n1
    S_min = S
    D = einsum('nj,jk,nk->n', X, S, X)
    # D = np.fromiter(
    #         (((x @ S) @ x) for x in X), 'd', N)
    qval_min = maf.evaluate(D)
    W = maf.gradient(D)

    for K in range(n_iter):
        S = (X.T @ diag(W)) @ X
        S = pinv(S)
        S /= det(S) ** n1
        D = einsum('nj,jk,nk->n', X, S, X)
        # D = np.fromiter(
        #         (((x @ S) @ x) for x in X), 'd', N)
        qval = maf.evaluate(D)

        stop = False
        if abs(qval - qval_min) < tol:
            stop = True

        if qval < qval_min:
            qval_min = qval
            S_min = S

        if stop:
            break

        W = maf.gradient(D)

    return S_min

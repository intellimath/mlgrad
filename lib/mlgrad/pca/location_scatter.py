import numpy as np
from numpy import diag, einsum, mean
from numpy.linalg import det, inv, pinv

def distance_center(X, c, /):
    Z = X - c
    # e = ones_like(c)
    # Z2 = (Z * Z) @ e #.sum(axis=1)    
    Z2 = einsum("ni,ni->n", Z, Z)
    return Z2

def location(X, /):
    return mean(X, axis=0)

def robust_location(XY, af, *, n_iter=1000, tol=1.0e-9, verbose=0):
    c = XY.mean(axis=0)
    c_min = c
    N = len(XY)

    # Z = XY - c
    # U = (Z * Z).sum(axis=1)
    U = distance_center(XY, c)
    af.fit(U)
    G = af.weights(U)
    s = s_min = af.u

    if verbose:
        print(s, c)

    for K in range(n_iter):
        c = XY.T @ G

        # Z = XY - c
        # U = (Z * Z).sum(axis=1)
        U = distance_center(XY, c)
        s = af.evaluate(U)
        G = af.gradient(U)
        
        # print(S, c)
        
        if K > 0 and s < s_min:
            s_min = s
            c_min = c
            if verbose:
                print('*', s, c)
        
        if K > 0 and abs(s - s_min) < tol:
            break

    if verbose:
        print(f"K: {K}")

    return c_min


def scatter_matrix(X):
    return X.T @ X / len(X)

def robust_scatter_matrix(X, maf, tol=1.0e-8, n_iter=100, verbose=True, return_qvals=False):
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

    qvals = [qval_min]
    for K in range(n_iter):
        S = (X.T @ diag(W)) @ X
        S = pinv(S)
        S /= det(S) ** n1
        D = einsum('nj,jk,nk->n', X, S, X)
        # D = np.fromiter(
        #         (((x @ S) @ x) for x in X), 'd', N)
        qval = maf.evaluate(D)
        qvals.append(qval)

        stop = False
        if abs(qval - qval_min) < tol:
            stop = True

        if qval < qval_min:
            qval_min = qval
            S_min = S
            if verbose:
                print(S, qval)

        if stop:
            break

        W = maf.gradient(D)

    if verbose:
        print(f"K: {K}")

    if return_qvals:
        return S_min, qvals, qval_min
    else:
        return S_min

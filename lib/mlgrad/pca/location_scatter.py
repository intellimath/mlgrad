import numpy as np
from numpy import diag, einsum, einsum_path, mean
from numpy.linalg import det, inv, pinv
from sys import float_info

def distance_center(X, c, /):
    Z = X - c
    # e = ones_like(c)
    # Z2 = (Z * Z) @ e #.sum(axis=1)    
    Z2 = einsum("ni,ni->n", Z, Z)
    return np.sqrt(Z2)

def location(X, /):
    return X.mean(axis=0)

def robust_location(X, af, *, n_iter=1000, tol=1.0e-8, verbose=0):
    c = X.mean(axis=0)
    c_min = c
    N = len(X)

    # Z = X - c
    # U = (Z * Z).sum(axis=1)
    Z = X - c

    path, _ = einsum_path("ni,ni->n", Z, Z, optimize='optimal')
    U = einsum("ni,ni->n", Z, Z, optimize=path)

    s = s_min = af.evaluate(U)
    s_min_prev = 0
    G = af.gradient(U)
    # print('*', s, G)

    if verbose:
        print(s, c)

    for K in range(n_iter):
        s_prev = s

        c = X.T @ G

        # Z = X - c
        # U = (Z * Z).sum(axis=1)
        Z = X - c
        U = einsum("ni,ni->n", Z, Z, optimize=path)
        # U = distance_center(XY, c)
        # print(U)
        s = af.evaluate(U)
        G = af.gradient(U)
        # print('**', s, G)

        # print(S, c)

        if s <= s_min:
            s_min_prev = s_min
            s_min = s
            c_min = c
            if verbose:
                print('*', s, c)
        
        if abs(s_prev - s) / (1 + abs(s_min)) < tol:
            break
        if abs(s_min - s_min_prev) / (1 + abs(s_min)) < tol:
            break

    if verbose:
        print(f"K: {K}")

    return c_min

def location_l1(X, *, n_iter=1000, tol=1.0e-9, verbose=0):
    c = X.mean(axis=0)
    c_min = c
    N = len(X)

    # Z = X - c
    # U = (Z * Z).sum(axis=1)
    Z = X - c

    path, _ = einsum_path("ni,ni->n", Z, Z, optimize='optimal')
    U = einsum("ni,ni->n", Z, Z, optimize=path)
    U = np.sqrt(U)

    s = s_min = U.mean()
    G = 1.0 / U
    G /= G.sum()
    # print('*', s, G)

    if verbose:
        print(s, c)

    for K in range(n_iter):
        s_prev = s
        c = X.T @ G

        # Z = X - c
        # U = (Z * Z).sum(axis=1)
        Z = X - c
        U = einsum("ni,ni->n", Z, Z, optimize=path)
        U = np.sqrt(U)
        # U = distance_center(XY, c)

        s = U.mean()
        G = 1.0 / U
        G /= G.sum()
        
        # print('**', s, G)

        # print(S, c)

        if s < s_min:
            s_min = s
            c_min = c
            if verbose:
                print('*', s, c)
        
        if abs(s_prev - s) / (1 + abs(s_min)) < tol:
            break

    if verbose:
        print(f"K: {K}", s_min, c_min)

    return c_min

def scatter_matrix(X):
    return X.T @ X / len(X)

def robust_scatter_matrix(X, maf, tol=1.0e-8, n_iter=100, verbose=False, return_qvals=False):
    N = len(X)
    S = X.T @ X
    n1 = 1.0 / S.shape[0]
    S /= det(S) ** n1
    S = inv(S)
    S_min = S
    path, _ = einsum_path('nj,jk,nk->n', X, S, X, optimize='optimal')
    D = einsum('nj,jk,nk->n', X, S, X, optimize=path)
    # D = np.fromiter(
    #         (((x @ S) @ x) for x in X), 'd', N)
    qval_min = maf.evaluate(D)
    qval_min_prev = float_info.max
    W = maf.weights(D)
    path2, _ = einsum_path('nj,n,nk->jk', X, W, X, optimize='optimal')

    qvals = [qval_min]
    for K in range(n_iter):
        # S = (X.T @ diag(W)) @ X
        S = einsum('nj,n,nk->jk', X, W, X, optimize=path2)
        S /= det(S) ** n1
        S = inv(S)
        D = einsum('nj,jk,nk->n', X, S, X, optimize=path)
        # D = np.fromiter(
        #         (((x @ S) @ x) for x in X), 'd', N)
        qval = maf.evaluate(D)
        W = maf.weights(D)
        qvals.append(qval)

        stop = False
        if abs(qval - qval_min) / (1 + abs(qval_min)) < tol:
            stop = True
        if abs(qval_min_prev - qval_min) / (1 + abs(qval_min)) < tol:
            stop = True

        if qval <= qval_min:
            qval_min_prev = qval_min
            qval_min = qval
            S_min = S
            if verbose:
                print(S, qval)
        
        if stop:
            break

    if verbose:
        print(f"K: {K}")

    if return_qvals:
        return S_min, qvals
    else:
        return S_min

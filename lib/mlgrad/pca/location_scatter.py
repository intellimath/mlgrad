import numpy as np
from numpy import diag, einsum, einsum_path, mean, average
from numpy.linalg import det, inv, pinv
from sys import float_info

import mlgrad.inventory as inventory

def distance_center(X, c, /):
    Z = X - c
    # e = ones_like(c)
    # Z2 = (Z * Z) @ e #.sum(axis=1)    
    Z2 = einsum("ni,ni->n", Z, Z)
    return np.sqrt(Z2)

def location(X, weights=None, /):
    if weights is None:
        return X.mean(axis=0)
    else:
        return average(X, axis=0, weights=weights)

def robust_location(X, af, *, n_iter=1000, tol=1.0e-6, verbose=0):
    c = X.mean(axis=0)
    c_min = c
    N = len(X)

    Xc = X - c
    path, _ = einsum_path("ni,ni->n", Xc, Xc, optimize='optimal')
    Z = einsum("ni,ni->n", Xc, Xc, optimize=path)

    s = s_min = af.evaluate(Z)
    s_min_prev = s_min * 10
    Q = 1 + abs(s_min)

    W = af.gradient(Z)

    if verbose:
        print(s, c)

    for K in range(n_iter):
        s_prev = s

        c = (W @ X) / W.sum()

        Xc = X - c
        Z = einsum("ni,ni->n", Xc, Xc, optimize=path) # Z = (Xc * Xc).sum(axis=1)
        s = af.evaluate(Z)
        W = af.gradient(Z)

        if s < s_min:
            s_min_prev = s_min
            s_min = s
            c_min = c
            if verbose:
                print('*', s, c)
            Q = 1 + abs(s_min)
        elif s < s_min_prev:
            s_min_prev = s
        
        if abs(s_prev - s) / Q < tol:
            break
        if abs(s - s_min_prev) / Q < tol:
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

def robust_location_scatter(X, maf, tol=1.0e-6, n_iter=1000, verbose=False, qvals=None):
    N, n = X.shape
    c = location(X)
    Xc = X - c
    S = Xc.T @ Xc / N
    S = inv(S)
    S_min = S
    c_min = c
    # path, _ = einsum_path('nj,jk,nk->n', Xc, S, Xc, optimize='optimal')
    # D = einsum('nj,jk,nk->n', Xc, S, Xc, optimize=path)
    D = inventory.mahalanobis_norm(S, Xc)
    # D = np.fromiter(
    #         (((x @ S) @ x) for x in X), 'd', N)
    qval_min = qval = maf.evaluate(D) - np.log(det(S))
    qval_min_prev = float_info.max / 100
    Q = 1 + abs(qval_min)
    W = maf.gradient(D)
    # path2, _ = einsum_path('nj,n,nk->jk', X, W, X, optimize='optimal')

    if qvals is not None:
        qvals.append(qval)

    m = 0
    for K in range(n_iter):
        qval_prev = qval
        c = np.average(X, axis=0, weights=W)
        Xc = X - c
        S = (Xc.T @ diag(W)) @ Xc
        # ### S = einsum('nj,n,nk->jk', Xc, W, Xc, optimize=path2)
        # S = inventory.scatter_matrix_weighted(Xc, W)
        # S /= det(S) ** n1
        S = inv(S)
        # ### D = einsum('nj,jk,nk->n', Xc, S, Xc, optimize=path)
        # # D = np.fromiter(
        # #         (((x @ S) @ x) for x in X), 'd', N)
        D = inventory.mahalanobis_norm(S, Xc)
        qval = maf.evaluate(D) - np.log(det(S))
        W = maf.gradient(D)

        if qvals is not None:
            qvals.append(qval)

        stop = False
        if abs(qval - qval_prev) / Q < tol:
            stop = True
        elif abs(qval - qval_min) / Q < tol:
            stop = True
        elif abs(qval - qval_min_prev) / Q < tol:
            if m >= 3:
                stop = True
            else:
                m += 1

        if qval < qval_min:
            qval_min_prev = qval_min
            qval_min = qval
            S_min = S
            c_min = c
            if verbose:
                print(qval, c, "\n", S)
            Q = 1 + abs(qval_min)
            m = 0
        elif qval < qval_min_prev:
            qval_min_prev = qval

        
        if stop:
            break

    if verbose:
        print(f"K: {K}")

    # D = np.fromiter(
    #         (((x @ S_min) @ x) for x in X), 'd', N)
    # D = einsum('nj,jk,nk->n', X, S_min, X, optimize=path)
    # maf.evaluate(D)
    # W = maf.gradient(D)
    # d = np.sqrt(n / (W @ D))

    return c_min, S_min

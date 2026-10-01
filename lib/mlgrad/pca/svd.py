import numpy as np
from math import sqrt

def find_uv(S, n_iter=200, tol=1.0e-9, verbose=0):
    nr, nc = S.shape
    u = 2*np.random.rand(nr)-1
    u /= sqrt(u @ u)
    v = 2*np.random.rand(nc)-1
    v /= sqrt(v @ v)
    lam = u @ (S @ v)
    if lam < 0:
        v = -v
        lam = -lam

    for K in range(n_iter):
        lam_prev = lam

        u1 = S @ v
        v1 = S.T @ u
        u = u1 / sqrt(u1 @ u1)
        v = v1 / sqrt(v1 @ v1)
        lam = u @ (S @ v)
        if lam < 0:
            v = -v
            lam = -lam

        if abs(lam - lam_prev) / (1 + abs(lam)) < tol:
            break

    # print(K+1)
    return u, v, lam

def find_uv_all(S, m=None, *, n_iter=200, tol=1.0e-9, verbose=False):
    nr, nc = S.shape
    mm = min(nr, nc, m)

    U = []
    Vt = []
    L = []

    S = S.copy()
    for i in range(mm):
        u, v, lam = find_uv(S, n_iter=n_iter, tol=tol, verbose=verbose)
        if lam == 0:
            break
        S -= lam * np.outer(u, v)
        L.append(lam)
        U.append(u)
        Vt.append(v)

    U = np.array(U)
    Vt = np.array(Vt)
    L = np.array(L)

    return U, Vt, L


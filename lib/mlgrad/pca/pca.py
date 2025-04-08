 #
# PCA
#

from sys import float_info
import numpy as np

from mlgrad.pca._pca import _find_pc
from mlgrad.smooth import whittaker_smooth
from mlgrad.pca.location_scatter import location, location_l1, robust_location

einsum = np.einsum
sqrt = np.sqrt
isnan = np.isnan
fromiter = np.fromiter

def distance_line(X, a, /):
    # e = ones_like(a)
    # XX = (X * X) @ e #.sum(axis=1)
    XX = einsum("ni,ni->n", X, X, optimize=True)
    Z = X @ a
    Z = XX - Z * Z
    Z[Z<0] = 0
    return sqrt(Z)

def score_distance(X, A, L, /):
    S = np.zeros(len(X), 'd')
    for a, l in zip(A, L):
        V = X @ a
        S += V * V / l
    return S

def project_line(X, a, /):
    return X @ a

def project(X, a, /):
    Xa = np.array([(x @ a) * a for x in X])
    # Xa = einsum("ni,i,j->nj", X, a, a, optimize=True)
    return X - Xa

# def project0(X, a, /):
#     Xa1 = einsum("ni,i,j->nj", X, a, a)
#     Xa2 = np.fromiter(((x @ a) * a for x in X), len(X), 'd')
#     return Xa1, Xa2

def total_regression(X, *, a0 = None, weights=None, n_iter=200, tol=1.0e-6, verbose=0):
    N = len(X)
    if weights is None:
        S = X.T @ X / N
    else:
        S = (X.T * weights) @ X
    a, L =  _find_pc(S, a0=a0, n_iter=n_iter, tol=tol, verbose=verbose) 
    return a, L

def find_pc(X, *, a0 = None, weights=None, n_iter=100, tol=1.0e-4, verbose=0):
    if weights is None:
        N = len(X)
        S = X.T @ X
    else:
        S = einsum("in,n,nj->ij", X.T, weights, X, optimize=True)
        # S = (X.T * weights) @ X
    a, L =  _find_pc(S, a0=a0, n_iter=n_iter, tol=tol, verbose=0)
    if verbose:
        print("*", L)
    return a, L

def find_pc_ss(X, *, a0 = None, n_iter=100, tol=1.0e-4, verbose=0):
    N = len(X)
    Xs = X.copy()
    for x in Xs:
        x[:] = x / abs(x).sum()
    S = Xs.T @ Xs
    a, L =  _find_pc(S, a0=a0, n_iter=n_iter, tol=tol, verbose=0)
    if verbose:
        print("*", L)
    return a, L

def find_pc_smoothed(X, *, a0=None, tau=1.0, weights=None, n_iter=100, tol=1.0e-4, verbose=0):
    N, m = X.shape
    if weights is None:
        S = X.T @ X
    else:
        # S = einsum("in,n,nj->ij", X.T, weights, X, optimize=True)
        S = (X.T * weights) @ X

    D = np.diff(np.eye(m, dtype="d"), 2)
    D2 = D @ D.T
    # print(S.shape, D2.shape)
    S -= tau * D2

    a, L =  _find_pc(S, a0=a0, n_iter=n_iter, tol=tol, verbose=verbose)
    return a, L

# def _find_pc(S, *, a0 = None, n_iter=1000, tol=1.0e-6, verbose=0):
#     if a0 is None:
#         a = np.random.random(S.shape[0])
#     else:
#         a = a0

#     np_abs = np.abs
#     np_sqrt = np.sqrt
#     np_sign = np.sign

#     a /= np.sqrt(a @ a)
    
#     for K in range(n_iter):
#         S_a = S @ a
#         L = S_a @ a
#         a1 = S_a / L
#         a1 /= np_sqrt(a1 @ a1)

#         if abs(a1 - a).max() / (1 + abs(a1).min()) < tol:
#             a = a1
#             break

#         a = a1

#     K += 1
#     if verbose:
#         print("K:", K, L, a)
            
#     S_a = S @ a
#     L = (S_a @ a) / (a @ a)
#     return a, L

def find_rho_pc(X, rho_func, *, a0=None, n_iter=100, tol=1.0e-6, verbose=0):
    N, n = X.shape

    np_abs = np.abs
    np_sqrt = np.sqrt
    
    if a0 is None:
        a0 = np.random.random(n)
    else:
        a0 = a0

    a = a_min = a0 / np.sqrt(a0 @ a0)
    XX = (X * X).sum(axis=1)

    Z = X @ a
    Z = rho_func.evaluate_array(XX - Z*Z)
    
    sz = sz_min = Z.mean()
    G = rho_func.derivative_array(Z)
    G /= G.sum()
    L_min = 0

    complete = False
    for K in range(n_iter):
        sz_prev = sz

        S = (X.T * G) @ X

        a1, L = _find_pc(S, a0=a, tol=tol, verbose=verbose)

        Z = X @ a1
        Z = rho_func.evaluate_array(XX - Z*Z)
        
        sz = Z.mean()
        G = rho_func.derivative_array(Z)
        G /= G.sum()
        
        if sz < sz_min:
            sz_min = sz
            a_min = a1
            L_min = L
            if verbose:
                print('*', sz, L, a)

        if abs(sz_prev - sz) / (1 + abs(sz_min)) < tol:
            break

        a = a1

    K += 1
    if verbose:
        print(f"K: {K}", sz_min, a_min, L_min)

    return a_min, L_min

def find_robust_pc(X, qf, *, a0=None, n_iter=100, tol=1.0e-6, verbose=0):
    N, n = X.shape

    if a0 is None:
        a0 = np.random.random(n)
    else:
        a0 = a0

    a = a_min = a0 / np.sqrt(a0 @ a0)
    XX = (X * X).sum(axis=1)

    _Z = X @ a
    Z = XX - _Z * _Z
    
    sz_min = sz = qf.evaluate(Z)
    sz_min_prev = float_info.max / 10
    sz_prev = float_info.max / 10
    G = qf.gradient(Z)
    L_min = 0

    np_abs = np.abs
    np_sqrt = np.sqrt

    complete = False
    for K in range(n_iter):

        S = (X.T @ np.diag(G)) @ X
        # S = einsum("in,n,nj->ij", X.T, G, X, optimize=True)

        a1, L = _find_pc(S, a0=a, tol=tol, verbose=0)

        _Z = X @ a1
        ZZ = XX - _Z * _Z
        
        sz = qf.evaluate(ZZ)
        G = qf.gradient(ZZ)

        if abs(sz - sz_min) / (1 + abs(sz_min)) < tol:
            complete = True
        # if abs(sz_min_prev - sz_min) / (1 + abs(sz_min)) < tol:
        #     complete = True

        # if abs(a1 - a_min).max()  / (1 + abs(a1).min()) < tol:
        #     complete = True
        
        if sz <= sz_min:
            sz_min_prev = sz_min
            sz_min = sz
            a_min = a1
            L_min = L
            if verbose == 2:
                print('*', sz, L) #, a1)

        if complete:
            break

        a = a1

    K += 1
    if verbose:
        print(f"K: {K}", sz_min)

    return a_min, L_min

def find_pc_l1(X, *, a0=None, n_iter=200, tol=1.0e-6, verbose=0):
    np_abs = np.abs
    np_sqrt = np.sqrt

    N, n = X.shape

    if a0 is None:
        a0 = np.random.random(n)
    else:
        a0 = a0

    a = a_min = a0 / np_sqrt(a0 @ a0)
    XX = (X * X).sum(axis=1)

    Z = X @ a
    Z1 = np_sqrt(np_abs(XX - Z * Z))
    sz = sz_min = Z1.mean()
    
    G = 1. / Z1
    G /= G.sum()
    L_min = 0

    for K in range(n_iter):
        sz_prev = sz

        S = (X.T * G) @ X

        a1, L = _find_pc(S, a0=a, n_iter=200, tol=tol, verbose=verbose)

        Z = X @ a1
        Z1 = np_sqrt(np_abs(XX - Z * Z))

        G = 1. / Z1
        G /= G.sum()
        sz = Z1.mean()

        if sz < sz_min:
            # Z1_min = Z1
            sz_min = sz
            a_min = a1
            L_min = L
            if verbose:
                print('*', sz, L, a)

        if abs(sz_prev - sz) / (1 + sz_min) < tol:
            break

        a = a1

    K += 1
    if verbose:
        print(f"K: {K}", sz_min, a_min, L_min)

    return a_min, L_min

def project(X, a, /):
    Xa = np.array([(x @ a) * a for x in X])
    return X - Xa

def transform(X, G):
    """
    X: исходная матрица
    G: матрица, столбцы которой суть главные компоненты
    """
    XG = X @ G
    Us = []
    for xg in XG:
        u = list(sum((xg_i*G_i for xg_i, G_i in zip(xg, G))))
        Us.append(u)
    U = np.array(Us)
    return U

def _find_pc_all(S, m):
    n = S.shape[0]
    S1 = S.copy()
    A = np.empty((m, n), dtype="d")
    Ls = np.empty(m, "d")
    for j in range(m):
        a, L = _find_pc(S1)
        A[j,:] = a
        L[j] = L
        S1 -= L*np.outer(a, a)
    return A, L

def _find_pc_all(S, m=None):
    n = S.shape[0]
    if m is None:
        n = m
    
    As = np.empty((m, n), dtype="d")
    Ls = np.empty(m, "d")
    # AA = np.empty((n, n), dtype="d")
    
    outer = np.outer
    for j in range(m):
        a, L = _find_pc(S)
        As[j,:] = a
        Ls[j] = L
        AA = outer(a, a)
        S = S - L * AA
    return As, Ls

def find_pc_all(X0, n=None, *, weights=None, verbose=False, return_us=True):
    N, m = X0.shape
    if n is None:
        n = m
    elif n > m:
        raise RuntimeError(f"n={n} greater X.shape[1]={m}")

    As = np.empty((n,m), "d")
    if return_us:
        Us = np.empty((N,n), "d")
    Ls = np.empty(n, "d")
    
    X = X0
    for i in range(n):
        a, L = find_pc(X, weights=weights, verbose=verbose)
        if return_us:
            U = project_line(X0, a)
        X = project(X, a)
        Ls[i] = L
        As[i,:] = a
        if return_us:
            Us[:,i] = U
    
    if return_us:
        return As, Ls, Us
    else:
        return As, Ls

def find_loc_and_pc(X, n=None, *, weights=None, verbose=False):
    c = location(X)
    Xc = X - c
    As, Ls = find_pc_all(Xc, n=n, weights=weights, verbose=verbose, return_us=False)
    return c, As, Ls

def find_pc_ss_all(X0, n=None, *, weights=None, verbose=False, return_us=True):
    N, m = X0.shape
    if n is None:
        n = m
    elif n > m:
        raise RuntimeError(f"n={n} greater X.shape[1]={m}")

    As = np.empty((n,m), "d")
    if return_us:
        Us = np.empty((N,n), "d")
    Ls = np.empty(n, "d")
    
    X = X0
    for i in range(n):
        a, L = find_pc_ss(X, verbose=verbose)
        if return_us:
            U = project_line(X0, a)
        X = project(X, a)
        Ls[i] = L
        As[i,:] = a
        if return_us:
            Us[:,i] = U
    
    if return_us:
        return As, Ls, Us
    else:
        return As, Ls

def find_loc_and_pc_ss(X, n=None, *, weights=None, verbose=False):
    c = location_l1(X)
    Xc = X - c
    As, Ls = find_pc_ss_all(Xc, n=n, verbose=verbose, return_us=False)
    return c, As, Ls
        

def find_pc_smoothed_all(X0, n=None, tau=1.0, *, weights=None, verbose=False, return_us=True):
    Ls = []
    As = []
    Us = []

    _n = X0.shape[1]
    if n is None:
        n = _n
    elif n > _n:
        raise RuntimeError(f"n={n} greater X.shape[1]={_n}")

    X = X0
    for i in range(n):
        a, L = find_pc_smoothed(X, tau=tau, weights=weights, verbose=verbose)
        U = project_line(X0, a)
        X = project(X, a)
        Ls.append(L)
        As.append(a)
        if return_us:
            Us.append(U)
    Ls = np.array(Ls)
    As = np.array(As)
    if return_us:
        Us = np.array(Us)

    if return_us:
        return As, Ls, Us
    else:
        return As, Ls
    
def find_pc_l1_all(X0, n=None, verbose=False):
    Ls = []
    As = []
    Us = []

    _n = X0.shape[1]
    if n is None:
        n = _n
    elif n > _n:
        raise RuntimeError(f"n={n} greater X.shape[1]={_n}")

    X = X0
    for i in range(n):
        if verbose:
            print(f"*** {i} ***")
        a, L = find_pc_l1(X, verbose=verbose)
        U = project_line(X0, a)
        X = project(X, a)
        Ls.append(L)
        As.append(a)
        Us.append(U)
    Ls = np.array(Ls)
    As = np.array(As)
    Us = np.array(Us)

    return As, Ls, Us

# def find_robust_pc_all2(X, wma, n=None, *, n_iter=200, tol=1.0e-6, verbose=0): 
#     c = location(X)
#     X0 = X - c
#     As, Ls, Us = rfind_pc_all(X0, n=n)
#     As = np.array(As)
    
#     XX0 = (X0 * X0).sum(axis=1)
    
#     U = X0 @ As
#     Z = XX0 - (U * U).sum(axis=1)
    
#     sz_min = sz = wma.evaluate(Z)
#     for K in range(n_iter):
#         sz_prev = sz
#         G = wma.gradient(Z)
#         c = np.average(Z, weights=G)
#         X0 - c
#         As, Ls, Us = find_pc_all(X0, n=n, weights=G)
#         As = np.array(As)
    
#         U = X0 @ As
#         Z = XX0 - (U * U).sum(axis=1)
        
#         sz = wma.evaluate(Z)

#         if sz < sz_min:
#             # Z1_min = Z1
#             sz_min = sz
#             As_min = As
#             Ls_min = Ls
#             Us_min = Us
#             if verbose:
#                 print('*', sz, Ls, As)

#         if abs(sz_prev - sz) / (1 + sz_min) < tol:
#             break

#     return As_min, Ls_min, Us_min
        
def find_robust_pc_all(X0, wma, n=None, *, verbose=False, return_us=True):
    N, m = X0.shape
    if n is None:
        n = m
    elif n > m:
        raise RuntimeError(f"n={n} greater X.shape[1]={m}")

    As = np.empty((n,m), "d")
    if return_us:
        Us = np.empty((N,n), "d")
    Ls = np.empty(n, "d")

    X = X0
    for i in range(n):
        a, L = find_robust_pc(X, wma, verbose=verbose)
        if return_us:
            U = project_line(X0, a)
        X = project(X, a)
        Ls[i] = L
        As[i,:] = a
        if return_us:
            Us[:,i] = U

    if return_us:
        return As, Ls, Us
    else:
        return As, Ls        

def find_rho_pc_all(X0, rho_func, n=None, *, verbose=False, return_us=True):
    Ls = []
    As = []
    Us = []
    _n = X0.shape[1]
    if n is None:
        n = _n
    elif n > _n:
        raise RuntimeError(f"n={n} greater X.shape[1]={_n}")
    X = X0
    for i in range(n):
        a, L = find_rho_pc(X, rho_func, verbose=verbose)
        if return_us:
            U = project_line(X, a)
        X = project(X, a)
        Ls.append(L)
        As.append(a)
        if return_us:
            Us.append(U)
    Ls = np.array(Ls)
    As = np.array(As)
    if return_us:
        Us = np.array(Us)
    Ls = np.array(Ls)
    if return_us:
        return As, Ls, Us
    else:
        return As, Ls

def find_robust_loc_and_pc(X, wma, n=None, *, verbose=False):
    c = robust_location(X, wma)
    Xc = X - c
    As, Ls = find_robust_pc_all(Xc, wma, n=n, verbose=verbose, return_us=False)
    return c, As, Ls

# def pca(data, numComponents=None):
#     """Principal Components Analysis

#     From: http://stackoverflow.com/a/13224592/834250

#     Parameters
#     ----------
#     data : `numpy.ndarray`
#         numpy array of data to analyse
#     numComponents : `int`
#         number of principal components to use

#     Returns
#     -------
#     comps : `numpy.ndarray`
#         Principal components
#     evals : `numpy.ndarray`
#         Eigenvalues
#     evecs : `numpy.ndarray`
#         Eigenvectors
#     """
#     m, n = data.shape
#     data -= data.mean(axis=0)
#     R = np.cov(data, rowvar=False)
#     # use 'eigh' rather than 'eig' since R is symmetric,
#     # the performance gain is substantial
#     evals, evecs = np.linalg.eigh(R)
#     idx = np.argsort(evals)[::-1]
#     evecs = evecs[:,idx]
#     evals = evals[idx]
#     if numComponents is not None:
#         evecs = evecs[:, :numComponents]
#     # carry out the transformation on the data using eigenvectors
#     # and return the re-scaled data, eigenvalues, and eigenvectors
#     return np.dot(evecs.T, data.T).T, evals, evecs


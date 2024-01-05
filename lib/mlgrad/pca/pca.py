 #
# PCA
#

import numpy as np
einsum = np.einsum
sqrt = np.sqrt
isnan = np.isnan
fromiter = np.fromiter

def distance_line(X, a, /):
    # e = ones_like(a)
    # XX = (X * X) @ e #.sum(axis=1)
    XX = einsum("ni,ni->n", X, X)
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
    Xa = (X @ a).reshape(-1,1) * X
    # Xa = np.fromiter(((x @ a) * a for x in X), len(X), 'd')
    return X - Xa

def total_regression(X, *, a0 = None, weights=None, n_iter=200, tol=1.0e-6, verbose=0):
    N = len(X)
    if weights is None:
        S = X.T @ X / N
    else:
        S = (X.T * weights) @ X
    a, L =  _find_pc(S, a0=a0, n_iter=n_iter, tol=tol, verbose=verbose) 
    return a, L

def find_pc(X, *, a0 = None, weights=None, n_iter=200, tol=1.0e-6, verbose=0):
    N = len(X)
    if weights is None:
        S = X.T @ X / N
    else:
        S = (X.T * np.diag(weights)) @ X
    a, L =  _find_pc(S, a0=a0, n_iter=n_iter, tol=tol, verbose=verbose) 
    return a, L

def _find_pc(S, *, a0 = None, n_iter=1000, tol=1.0e-6, verbose=0):
    if a0 is None:
        a = np.random.random(S.shape[0])
    else:
        a = a0

    np_abs = np.abs
    np_sqrt = np.sqrt
    np_sign = np.sign

    a /= np.sqrt(a @ a)
    
    for K in range(n_iter):
        S_a = S @ a
        L = S_a @ a
        a1 = S_a / L
        a1 /= np_sqrt(a1 @ a1)
                
        if abs(a1 - a).max() < tol:
            a = a1
            break

        a = a1

    K += 1
    if verbose:
        print("K:", K, L, a)
            
    S_a = S @ a
    L = (S_a @ a) / (a @ a)
    return a, L

def find_rho_pc(X, rho_func, *, a0=None, n_iter=1000, tol=1.0e-6, verbose=0):
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
    
    SZ_min = Z.mean()
    G = rho_func.derivative_array(Z)
    G /= G.sum()
    L_min = 0

    complete = False
    for K in range(n_iter):

        S = (X.T * G) @ X

        a1, L = _find_pc(S, a0=a, tol=tol, verbose=verbose)

        Z = X @ a1
        Z = rho_func.evaluate_array(XX - Z*Z)
        
        SZ = Z.mean()
        G = rho_func.derivative_array(Z)
        G /= G.sum()

        # if abs(SZ - SZ_min) / (1 + abs(SZ_min)) < tol:
        #     complete = True

        if abs(a1 - a_min).max() < tol:
            complete = True
        
        if SZ < SZ_min:
            SZ_min = SZ
            a_min = a1
            L_min = L
            if verbose:
                print('*', SZ, L, a)

        if complete:
            break

        a = a1

    K += 1
    if verbose:
        print(f"K: {K}")

    return a_min, L_min

def find_robust_pc(X, qf, *, a0=None, n_iter=1000, tol=1.0e-6, verbose=0):
    N, n = X.shape

    if a0 is None:
        a0 = np.random.random(n)
    else:
        a0 = a0

    a = a_min = a0 / np.sqrt(a0 @ a0)
    XX = (X * X).sum(axis=1)

    _Z = X @ a
    Z = XX - _Z * _Z
    
    SZ_min = qf.evaluate(Z)
    G = qf.gradient(Z)
    L_min = 0

    np_abs = np.abs
    np_sqrt = np.sqrt

    complete = False
    for K in range(n_iter):

        S = (X.T * G) @ X

        a1, L = _find_pc(S, a0=a, tol=tol, verbose=verbose)

        Z = X @ a1
        ZZ = XX - Z * Z
        
        SZ = qf.evaluate(ZZ)
        G = qf.gradient(ZZ)

        # if abs(SZ - SZ_min) / (1 + abs(SZ_min)) < tol:
        #     complete = True

        if abs(a1 - a_min).max() < tol:
            complete = True
        
        if SZ < SZ_min:
            SZ_min = SZ
            a_min = a1
            L_min = L
            if verbose:
                print('*', SZ, L, a)

        if complete:
            break

        a = a1

    K += 1
    if verbose:
        print(f"K: {K}")

    return a_min, L_min

# def find_pc_l1(X, *, a0=None, n_iter=200, tol=1.0e-6, verbose=0, l1=False, tau=0.001):
#     N, n = X.shape

#     if a0 is None:
#         a0 = np.random.random(n)
#     else:
#         a0 = a0

#     a = a_min = a0 / np_sqrt(a0 @ a0)
#     XX = (X * X).sum(axis=1)

#     Z = X @ a
#     Z1 = np_sqrt(XX - Z * Z)
#     SZ = SZ_min = Z1.sum()
    
#     G = 1. / Z1
#     L_min = 0

#     np_abs = np.abs
#     np_sqrt = np.sqrt

#     complete = False
#     for K in range(n_iter):

#         S = (X.T * G) @ X

#         a1, L = _find_pc(S, a0=a, n_iter=200, tol=tol, verbose=verbose, l1=l1, tau=tau)

#         Z = X @ a1
#         Z1 = np_sqrt(XX - Z * Z)    

#         G = 1. / Z1
#         SZ = Z1.sum()

#         if abs(SZ - SZ_min) / (1 + SZ_min) < tol:
#             complete = True

#         if SZ < SZ_min:
#             # Z1_min = Z1
#             SZ_min = SZ
#             a_min = a1
#             L_min = L
#             if verbose:
#                 print('*', SZ, L, a)

#         if complete:
#             break

#         a = a1

#     K += 1
#     if verbose:
#         print(f"K: {K}")

#     return a_min, L_min

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

def find_pc_all(X0, n=None):
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
        a, L = find_pc(X)
        U = project_line(X0, a)
        X = project(X, a)
        Ls.append(L)
        As.append(a)
        Us.append(U)
    Ls = np.array(Ls)
    return As, Ls, Us

def _find_robust_pc_all(X, wma, n=None, As=None, *, n_iter=200, tol=1.0e-6, verbose=0): 
    if As is None:
        return find_pc_all(X, n=n)
    N = len(X)
    S = X.T @ X / N
    a, L =  _find_pc(S, a0=a0, n_iter=n_iter, tol=tol, verbose=verbose) 
    return a, L

def find_robust_pc_all(X0, wma, n=None, verbose=False):
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
        a, L = find_robust_pc(X, wma, verbose=verbose)
        U = project_line(X0, a)
        X = project(X, a)
        Ls.append(L)
        As.append(a)
        Us.append(U)
    Ls = np.array(Ls)
    return As, Ls, Us

def find_rho_pc_all(X0, rho_func, n=None, verbose=False):
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
        U = project_line(X0, a)
        X = project(X, a)
        Ls.append(L)
        As.append(a)
        Us.append(U)
    Ls = np.array(Ls)
    return As, Ls, Us

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


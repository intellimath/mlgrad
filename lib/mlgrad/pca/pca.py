#
# PCA
#

import numpy as np
einsum = np.einsum

def find_center(X, /):
    return np.mean(X, axis=0)

def distance_center(X, c, /):
    # e = ones_like(c)
    Z = X - c
    # Z2 = (Z * Z) @ e #.sum(axis=1)    
    Z2 = einsum("ni,ni->n", Z, Z)
    return Z2

def distance_line(X, a, /):
    # e = ones_like(a)
    # XX = (X * X) @ e #.sum(axis=1)
    XX = einsum("ni,ni->n", X, X)
    Z = X @ a
    Z = XX - Z * Z
    return Z

def project_line(X, a, /):
    return X @ a

def find_rob_center(XY, af, *, n_iter=1000, tol=1.0e-9, verbose=0):
    c = XY.mean(axis=0)
    c_min = c
    N = len(XY)

    # Z = XY - c
    # U = (Z * Z).sum(axis=1)
    U = distance_center(XY, c)
    af.fit(U, None)
    G = af.weights(U)
    S = S_min = af.u

    if verbose:
        print(S, c)

    for K in range(n_iter):
        c = XY.T @ G

        # Z = XY - c
        # U = (Z * Z).sum(axis=1)
        U = distance_center(XY, c)
        af.fit(U, None)
        G = af.gradient(U)
        
        S0 = S
        S = af.u
        # print(S, c)
        
        if K > 0 and S < S_min:
            S_min = S
            c_min = c
            if verbose:
                print('*', S, c)
        
        if K > 0 and abs(S - S_min) < tol:
            break

    if verbose:
        print(f"K: {K}")

    return c_min

def find_pc(XY2, *, a0 = None, n_iter=1000, tol=1.0e-8, verbose=0):
    N, n = XY2.shape
    if a0 is None:
        a = np.random.random(n)
    else:
        a = a0

    S = XY2.T @ XY2 / N
    XX = (XY2 * XY2).sum(axis=1)
    
    np_abs = np.abs
    np_sqrt = np.sqrt
    
    for K in range(n_iter):
        L = ((S @ a) @ a) / (a @ a)
        a1 = (S @ a) / L
        a1 /= np_sqrt(a1 @ a1)
                
        if np_abs(a1 - a).max() < tol:
            break

        a = a1
        if verbose:
            print(L, S)

    # Z = XY2 @ a1
    # Z = XX - Z * Z
            
    return a, L

def find_rob_pc(X, qf, *, n_iter=1000, tol=1.0e-8, verbose=0):
    N, n = X.shape

    a0 = np.random.random(n)
    a = a_min = a0 / np.sqrt(a0 @ a0)
    XX = (X * X).sum(axis=1)
    # print(XX.shape)    

    Z = X @ a
    Z = Z_min = XX - Z * Z
    
    qf.fit(Z, None)
    SZ_min = qf.u
    G = qf.gradient(Z)
    L_min = 0

    np_abs = np.abs
    np_sqrt = np.sqrt
    
    for K in range(n_iter):

        S = (X.T * G) @ X

        L = ((S @ a) @ a) / (a @ a)
        a1 = (S @ a) / L
        a1 /= np_sqrt(a1 @ a1)
        
        Z = X @ a1
        Z = XX - Z * Z
        
        qf.fit(Z, None)
        SZ = qf.u
        G = qf.gradient(Z)

        if SZ < SZ_min:
            SZ_min = SZ
            a_min = a1
            L_min = L
            Z_min = Z
            if verbose:
                print('*', SZ, L, a)

        if np_abs(a1 - a).max() < tol:
            break

        a = a1
        
    if verbose:
        print(f"K: {K}")

    return a_min, L_min

def project(X, a, /):
    # Xa = (X @ a).reshape(-1,1) * X
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

def find_pc_all(X0):
    Ls = []
    As = []
    Us = []
    n = X0.shape[1]
    X = X0
    for i in range(n):
        a, L = find_pc(X)
        U = project_line(X0, a)
        X = project(X, a)
        Ls.append(L)
        As.append(L)
        Us.append(U)
    Ls = np.array(Ls)
    return As, Ls, Us
        
def find_rob_pc_all(X0, wma):
    Ls = []
    As = []
    Us = []
    n = X0.shape[1]
    X = X0
    for i in range(n):
        a, L = find_rob_pc(X, wma)
        U = project_line(X0, a)
        X = project(X, a)
        Ls.append(L)
        As.append(L)
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


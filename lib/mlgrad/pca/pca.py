import numpy as np
# import matplotlib.pyplot as plt
# import mlgrad.af as af
# import mlgrad.func as func
# import mlgrad.utils as utils


def find_center(XY):
    return mp.mean(XY, axis=1)

def find_rob_center(XY, af, n_iter=1000, tol=1.0e-9, verbose=0):
    c = XY.mean(axis=0)
    c_min = c
    N = len(XY)

    Z = XY - c
    U = (Z * Z).sum(axis=1)
    af.fit(U, None)
    G = af.gradient(U)
    S = S_min = af.u

    if verbose:
        print(S, c)

    for K in range(n_iter):
        c = XY.T @ G

        Z = XY - c
        U = (Z * Z).sum(axis=1)
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

    return K, c_min


def find_pc(XY2, a0 = None, n_iter=1000, tol=1.0e-8, verbose=0):
    N, n = XY2.shape
    if a0 is None:
        a = np.random.random(n)
    else:
        a = a0

    S = XY2.T @ XY2
    XX = np.sum(XY2*XY2, axis=1)

    np_abs = np.abs
    np_sqrt = np.sqrt
    
    for K in range(n_iter):
        L = ((S @ a) @ a) / (a @ a)
        a1 = (S @ a) / L
        a1 /= np_sqrt(a1 @ a1)
        
        Z = XY2 @ a1
        Z = XX - Z * Z
        
        if np_abs(a1 - a).max() < tol:
            break

        a = a1
        if verbose:
            print(L, S)

    return a, L, Z

def find_rob_pc(XY, qf, n_iter=1000, tol=1.0e-8, verbose=0):
    N, n = XY.shape

    a = a_min = np.random.random(n)
    XX = np.sum(XY*XY, axis=1)
    print(XX.shape)

    Z = XY @ a
    Z = Z_min = XX - Z * Z
    qf.fit(Z, None)
    SZ_min = qf.u
    G = qf.gradient(Z) * N

    np_abs = np.abs
    np_sqrt = np.sqrt
    
    for K in range(n_iter):

        S = (XY.T * G) @ XY

        L = ((S @ a) @ a) / (a @ a)
        a1 = (S @ a) / L
        a1 /= np_sqrt(a1 @ a1)
        
        Z = XY @ a1
        Z = XX - Z * Z
        qf.fit(Z, None)
        SZ = qf.u
        G = qf.gradient(Z) * N

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

    return a_min, L_min, Z_min
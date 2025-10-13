import numpy as np

def exclude_outliers(mod, X, Y, n):
    if len(X.shape) == 1:
        Y_p = mod.evaluate(X.reshape(-1,1))
    else:
        Y_p = mod.evaluate(X)
    E = np.abs(Y - Y_p)
    I = np.argsort(E)
    return np.ascontiguousarray(X[I[:-n]]), np.ascontiguousarray(Y[I[:-n]])

def exclude_tail(keyvals, X, n):
    I = np.argsort(keyvals)
    return np.ascontiguousarray(X[I[:-n]])

def get_outliers(mod, X, Y, n):
    if len(X.shape) == 1:
        Y_p = mod.evaluate_all(X[:,None])
    else:
        Y_p = mod.evaluate_all(X)
    E = np.abs(Y - Y_p)
    I = np.argsort(E)
    return np.ascontiguousarray(X[I[-n:]]), np.ascontiguousarray(Y[I[-n:]])
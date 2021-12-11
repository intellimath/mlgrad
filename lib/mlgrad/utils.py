import numpy as np

def exclude_outliers(mod, X, Y, n):
    if len(X.shape) == 1:
        Y_p = mod.evaluate_all(X[:,None])
    else:
        Y_p = mod.evaluate_all(X)
    I = np.argsort(np.abs(Y - Y_p))
    return X[I[:-n]], Y[I[:-n]]
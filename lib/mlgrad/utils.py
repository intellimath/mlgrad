import numpy as np

def array_exclude_outliers(X, Err, n):
    I = np.argsort(Err)
    return X[I[:-n]], X[I[-n:]]
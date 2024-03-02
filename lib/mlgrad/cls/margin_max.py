import numpy as np


class MarginMaximization:

    def __init__(self, func, tol=1.0e-9, n_iter=200):
        self.func = func
        self.tol = tol
        self.n_iter = n_iter
    #
    def fit(self, X, Y):
        func = self.func
        tol = self.tol

        YX = Y[:,1] * X

        N, n = X.shape
        w = np.random.random(n)
        c = 0

        for K in range(self.n_iter):
            
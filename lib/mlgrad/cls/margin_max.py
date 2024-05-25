import numpy as np
from sys import float_info
from math import log

class MarginMaximization2:

    def __init__(self, func, h=0.01, tol=1.0e-9, n_iter=1000, verbose=False):
        self.func = func
        self.h = h
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.c = None
        self.w = None
        # self.s = 1
    #
    def fit(self, X, Y):
        sqrt = np.sqrt
        outer = np.outer
        func = self.func
        tol = self.tol
        h = self.h
        verbose = self.verbose

        if self.w is None:
            N, n = X.shape
            w = 2*np.random.random(n)-1
            w /= np.sqrt(w @ w)
            self.w = w
        else:
            w = self.w

        if self.c is None:
            c = self.c = 0
        else:
            c = self.c

        # s = self.s = 1.0

        N = len(X)
        
        w_min = w.copy()
        c_min = c
        # s_min = s
        
        XT = X.T

        Xw = X @ w
        U = (Xw - c) * Y
        
        lval = lval_min = func.evaluate_array(U).sum()
        # lval_min_prev = float_info.max
        self.lvals = [lval]

        for K in range(self.n_iter):
            lval_prev = lval

            V = func.derivative_array(U) * Y

            g = (XT - outer(w, Xw)) @ V
            w -= h * g
            c += h * V.sum()
            w /= sqrt(w @ w)

            Xw = X @ w
            U = (Xw - c) * Y

            # Xw = X @ w
            # U = (Xw - c) * Y
            
            lval = func.evaluate_array(U).sum()
            self.lvals.append(lval)
            
            if lval < lval_min:
                # lval_min_prev = lval_min 
                lval_min = lval
                w_min = w.copy()
                c_min = c
                # s_min = s
                if verbose:
                    print("K:", K, "w:", w, "c:", c)
            
            if abs(lval_min - lval_prev) / (1 + abs(lval_min)) < tol:
                break

            # I = U < 0
            # UI = U[I]
            # s = (func.derivative_array(UI/s) @ UI) / I.sum()

            # U /= s

            # if abs(lval_min - lval_min_prev) / (1 + abs(lval_min)) < tol:
            #     break

        self.K = K
        self.w = w_min        
        self.c = c_min
        # self.s = s_min

    def evaluate(self, X):
        return X @ self.w - self.c
    #
    evaluate_all = evaluate

class MarginMaximization1:

    def __init__(self, func, h=0.01, tol=1.0e-9, n_iter=1000, verbose=False):
        self.func = func
        self.h = h
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.c = None
        self.w = None
        # self.s = 1
    #
    def fit(self, X, Y):
        sqrt = np.sqrt
        outer = np.outer
        sign = np.sign
        func = self.func
        tol = self.tol
        h = self.h
        verbose = self.verbose

        if self.w is None:
            N, n = X.shape
            w = 2*np.random.random(n)-1
            w /= np.sqrt(w @ w)
            self.w = w
        else:
            w = self.w

        if self.c is None:
            c = self.c = 0
        else:
            c = self.c

        w_min = w.copy()
        c_min = c
        # s_min = s
        
        XT = X.T

        Xw = X @ w
        U = (Xw - c) * Y
        
        lval = lval_min = func.evaluate_array(U).sum()
        # lval_min_prev = float_info.max
        self.lvals = [lval]

        for K in range(self.n_iter):
            lval_prev = lval

            V = func.derivative_array(U) * Y

            sign_w = sign(w)
            g = (XT - outer(sign_w, Xw)) @ V
            w -= h * g
            c += h * V.sum()
            w /= sign_w @ w

            Xw = X @ w
            U = (Xw - c) * Y
            
            lval = func.evaluate_array(U).sum()
            self.lvals.append(lval)

            if lval < lval_min:
                # lval_min_prev = lval_min
                lval_min = lval
                w_min = w.copy()
                c_min = c
                # s_min = s
                if verbose:
                    print("K:", K, "w:", w, "c:", c)
            
            if abs(lval_min - lval_prev) / (1 + abs(lval_min)) < tol:
                break
            # if abs(lval_min - lval_min_prev) / (1 + abs(lval_min)) < tol:
            #     break

        self.K = K
        self.w = w_min        
        self.c = c_min
        # self.s = s_min

    def evaluate(self, X):
        return X @ self.w - self.c
    #
    evaluate_all = evaluate

import numpy as np
from sys import float_info
from math import log

class MarginMaximization:

    def __init__(self, func, h=0.01, tol=1.0e-9, n_iter=1000, verbose=False):
        self.func = func
        self.h = h
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.c = None
        self.w = None
    #
    def fit(self, X, Y):
        sqrt = np.sqrt
        outer = np.outer
        func = self.func
        tol = self.tol
        h = self.h
        verbose = self.verbose

        if self.w is None:
            w = X[Y>0].mean(axis=0) - X[Y<0].mean(axis=0)
            w /= np.sqrt(w @ w)
            self.w = w
        else:
            w = self.w

        
        if self.c is None:
            c = self.c = (X[Y>0].mean(axis=0) + X[Y<0].mean(axis=0)) / 2
        else:
            c = self.c

        N = len(X)
        
        w_min = w.copy()
        c_min = c.copy()

        Xw = (X - c) @ w
        U = Xw * Y
        
        lval = lval_min = func.evaluate_sum(U)
        self.lvals = [lval]

        for K in range(self.n_iter):
            lval_prev = lval

            V = func.derivative_array(U)
            VY = V * Y

            g = VY @ (X - c) - (V @ U) * w
            w -= h * g / N
            c += h * VY.mean() * w 
            w /= sqrt(w @ w)

            Xw = (X - c) @ w
            U = Xw * Y
            
            lval = func.evaluate_sum(U)
            self.lvals.append(lval)
            
            if lval < lval_min:
                # lval_min_prev = lval_min
                lval_min = lval
                w_min = w.copy()
                c_min = c.copy()
                if verbose:
                    print("K:", K, "w:", w, "c:", c)
            
            if abs(lval - lval_prev) / (1 + abs(lval_min)) < tol:
                break

            # if abs(lval_min - lval_min_prev) / (1 + abs(lval_min)) < tol:
            #     break
        
        self.K = K
        self.w = w_min        
        self.c = c_min

    def evaluate(self, X):
        return (X - self.c) @ self.w
    #
    evaluate_all = evaluate


class MarginMaximization2:

    def __init__(self, func, h=0.01, tol=1.0e-9, n_iter=1000, verbose=False):
        self.func = func
        self.h = h
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.c = None
        self.w = None
    #
    def fit(self, X, Y):
        sqrt = np.sqrt
        outer = np.outer
        func = self.func
        tol = self.tol
        h = self.h
        verbose = self.verbose

        if self.w is None:
            w = X[Y>0].mean(axis=0) - X[Y<0].mean(axis=0)
            w /= np.sqrt(w @ w)
            self.w = w
        else:
            w = self.w

        
        if self.c is None:
            c = self.c = (X[Y>0].mean(axis=0) + X[Y<0].mean(axis=0)) / 2
        else:
            c = self.c
        
        w_min = w.copy()
        c_min = c.copy()

        Xc = X - c 
        Xw = Xc @ w
        U = Xw * Y
        
        lval = lval_min = func.evaluate_sum(U)
        # lval_min_prev = float_info.max / 1000
        self.lvals = [lval]

        for K in range(self.n_iter):
            lval_prev = lval

            V = func.derivative_array(U)
            # L = -U @ V
            V *= Y
            w = (Xc.T @ V) / (U @ V)
            w /= np.sqrt(w @ w)
    
            V = func.derivative_div_array(U)
            c = (X.T @ V) / V.sum()

            Xc = X - c 
            Xw = Xc @ w
            U = Xw * Y
            
            lval = func.evaluate_sum(U)
            self.lvals.append(lval)
            
            # if lval < lval_min:
            #     # lval_min_prev = lval_min
            #     lval_min = lval
            #     w_min = w.copy()
            #     c_min = c.copy()
            #     if verbose:
            #         print("K:", K, "w:", w, "c:", c)
            
            if abs(lval - lval_prev) / (1 + abs(lval_min)) < tol:
                break

            # if abs(lval_min - lval_min_prev) / (1 + abs(lval_min)) < tol:
            #     break            
        
        self.K = K
        # self.w = w_min        
        # self.c = c_min
        self.w = w
        self.c = c

        Xw = (X - self.c) @ self.w
        U = Xw * Y
        L = -U @ func.derivative_array(U)
        
        self.L = L

    def evaluate(self, X):
        return (X - self.c) @ self.w
    #
    evaluate_all = evaluate



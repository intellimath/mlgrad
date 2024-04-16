import numpy as np

class MarginMaximization:

    def __init__(self, func, tol=1.0e-9, n_iter=1000, n_iter_c=22, n_iter_w=22):
        self.func = func
        self.tol = tol
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_w = n_iter_w
        self.c = None
        self.w = None
    #
    def fit_w(self, X, Y):
        sqrt = np.sqrt
        func = self.func
        tol = self.tol
        c = self.c
        w = self.w
        w_min = w.copy()

        XT = X.T

        Xw = X @ w
        U = (Xw - c) * Y
        
        lval = lval_min = func.evaluate_array(U).mean()

        for K in range(self.n_iter_w):
            lval_prev = lval

            V = func.derivative_array(U) * Y

            w1 = (XT @ V) / (Xw @ V)
            w = w1 / sqrt(w1 @ w1)

            Xw = X @ w
            U = (Xw - c) * Y
            
            lval = func.evaluate_array(U).mean()
            self.lvals.append(lval)

            if lval < lval_min:
                lval_min = lval
                w_min = w.copy()
            
            if abs(lval - lval_prev) / (1 + abs(lval_min)) < tol:
                break

        print(K)
        self.w = w_min        
    #
    def fit_c(self, X, Y):
        func = self.func
        tol = self.tol
        c = c_min = self.c
        w = self.w
        
        Xw = X @ w
        U = (Xw - c) * Y

        lval= lval_min = func.evaluate_array(U).mean()

        for K in range(self.n_iter_c):
            lval_prev = lval

            V = func.derivative_div_array(U)
            c = (Xw @ V) / V.sum()

            # Xw = X @ w
            U = (Xw - c) * Y
            
            lval= func.evaluate_array(U).mean()
            self.lvals.append(lval)
            
            if lval < lval_min:
                lval_min = lval
                c_min = c
                
            if abs(lval - lval_prev) / (1 + abs(lval_min)) < tol:
                break

        print(K)
        self.c = c_min        
    #
    def fit(self, X, Y):
        func = self.func
        tol = self.tol

        if self.w is None:
            N, n = X.shape
            w = 2*np.random.random(n)-1
            w /= np.sqrt(w @ w)
            self.w = w
        # else:
        #     w = self.w

        if self.c is None:
            Xw = X @ self.w
            U = Xw * Y
            V = func.derivative_div_array(U)
            c = (V @ Xw) / V.sum()
            self.c = c
        # else:
        #     c = self.c

        Xw = X @ self.w
        U = (Xw - self.c) * Y
        
        lval= lval_min = func.evaluate_array(U).mean()
        self.lvals = lvals = [lval]
        self.lvals2 = [lval]
        
        for K in range(self.n_iter):
            lval_prev = lval

            self.fit_w(X, Y)
            self.fit_c(X, Y)

            Xw = X @ self.w
            U = (Xw - self.c) * Y

            lval = func.evaluate_array(U).mean()
            self.lvals2.append(lval)
            
            if abs(lval - lval_prev) / (1 + abs(lval_min)) < tol:
                break
    
        self.K = K

    def evaluate(self, X):
        return X @ self.w - self.c

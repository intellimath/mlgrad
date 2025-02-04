import numpy as np

def normalize(a):
    return a / np.sqrt(a @ a)

class CLS_PCA:
    #
    def __init__(self, func, n_iter=1000, tol=1.0e-8, h=0.1):
        self.func = func
        self.n_iter = n_iter
        self.tol = tol
        self.h = h
        self.A = None
        self.A0 = None
        self.b = None
    #
    def fit(self, X, Y, n_pc=-1):
        N, n = X.shape
        if n_pc < 0:
            n_pc = n

        if self.A is None:
            A = 2*np.random.random(size=(n_pc, n))-1
            for i in range(n_pc):
                A[i,:] = normalize(A[i])
            self.A = A
            self.A0 = A0 = 2*np.random.random(n_pc)-1
        else:
            A = self.A
            A0 = self.A0

        if self.b is None:
            b = self.b = 2*np.random.random(n_pc)-1
            b0 = self.b0 = 2*np.random.random(1)-1
        else:
            b = self.b
            b = self.b0

        h = self.h
        tol = self.tol

        Ones = np.ones((N,1), "d")
        X1 = np.concatenate((Ones, X), axis=1)

        XA = X @ A.T # (N,n) @ (n,n_pc) -> (N,n_pc)
        YY =  XA @ b # (N,n_pc) @ (n_pc) -> (N)
        U = YY * Y
        qval = qval_min = (self.func.evaluate_array(U) * Y).sum()
        qvals = [qval]

        A_min, b_min = A.copy(), b.copy()

        print(A, b)

        for K in range(self.n_iter):
            qval_prev = qval
            YFD = self.func.derivative_array(U) * Y
            XAA = np.einsum("kj,jn", XA, A, optimize=True)
            XA2 = X - XAA[:,1:]
            A[:,1:] -= h * np.einsum("k,kn,j", YFD, XA2, b[1:], optimize=True)
            b[0] -= h * YFD.sum()
            A[:,0] -= h * YFD.sum() * b[1:]

            for i in range(n_pc):
                A[i,1:] = normalize(A[i,1:])

            print(i, A, b)
            
            XA = X1 @ A.T # (N,n+1) @ (n+1,n_pc) -> (N,n_pc)
            YY =  XA @ b[1:] + b[0] # (N,n_pc) @ (n_pc) -> (N)
            U = YY * Y
            
            qval = (self.func.evaluate_array(U) * Y).sum()
            qvals.append(qval)

            if qval < qval_min:
                qval_min = qval
                A_min, b_min = A.copy(), b.copy()

            if abs(qval - qval_prev) / (1 + abs(qval_min)) < tol:
                break

        self.A = A_min
        self.b = b_min
        self.qvals = qvals
        self.K = K + 1

    def evaluate(self, X):
        N = len(X)
        Ones = np.ones((N,1), "d")
        X1 = np.concatenate((Ones, X), axis=1)

        XA = X1 @ self.A.T # (N,n+1) @ (n+1,n_pc) -> (N,n_pc)
        YY =  XA @ self.b[1:] + self.b[0] # (N,n_pc) @ (n_pc) -> (N)

        return YY
                
            
        
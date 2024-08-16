#
#
#

import numpy as np


cdef class WhittakerSmoother:
    #
    def __init__(self, funcs2.Func2 func=None, funcs2.Func2 func2=None, 
                 h=0.001, n_iter=1000, 
                 tol=1.0e-6, tau=10.0):
        if func is None:
            self.func = funcs2.SquareNorm()
        else:
            self.func = func
        if func2 is None: 
            self.func2 = funcs2.FuncDiff2(funcs.Square())
        else:
            self.func2 = func2
        self.n_iter = n_iter
        self.tol = tol
        self.h = h
        self.tau = tau
        self.Z = None
        self.qvals = None
    #
    def fit(self, double[::1] X, double[::1] W=None, double[::1] W2=None):
        cdef double h = self.h
        cdef double tau = self.tau
        cdef double tol = self.tol
        cdef funcs2.Func2 func = self.func
        cdef funcs2.Func2 func2 = self.func2
        cdef Py_ssize_t j, N = len(X)
        cdef averager.ArrayAverager avg
        cdef double[::1] Z = np.zeros(N, 'd')
        cdef double[::1] Z_min = np.zeros(N, 'd')
        cdef double[::1] E = np.zeros(N, 'd')
        cdef double[::1] G1 = np.zeros(N, 'd')
        cdef double[::1] G2 = np.zeros(N, 'd')
        cdef double[::1] grad = np.zeros(N, 'd')
        cdef double qval, qval_prev, qval_min, qval_min_prev
        cdef list qvals
        cdef int M = 0

        # avg = averager.ArrayAdaM2()
        # avg.init(N)

        if W is None:
            W = np.ones_like(X)
        # inventory.normalize(W)

        if W2 is None:
            W2 = np.ones_like(X)
        # inventory.normalize(W2)
        
        if self.Z is None:
            inventory.move(Z, X)
            # Z = X.copy()
        else:
            inventory.move(Z, self.Z)
            # Z = self.Z
        inventory.move(Z_min, Z)
        # Z_min = Z.copy()

        inventory.sub(E, X, Z)
        #  qval = func._evaluate_ex(E, W) + tau * func2._evaluate(Z)
        qval = func._evaluate(E) / tau + func2._evaluate(Z)
        qvals = [qval]
    
        qval_min = qval
        qval_min_prev = 2.0 * qval_min

        for K in range(self.n_iter):
            qval_prev = qval
            # Z_prev = Z.copy()

            # func._gradient_ex(E, G1, W) 
            # func2._gradient(Z, G2)
            func._gradient(E, G1) 
            func2._gradient(Z, G2)
            for j in range(N):
                grad[j] = -G1[j] / tau + G2[j]

            # avg.update(grad, h)
            
            for j in range(N):
                Z[j] -= h * grad[j]

            # inventory.isub(Z, avg.array_average)

            inventory.sub(E, X, Z)
            
            # qval = func._evaluate_ex(E, W) + tau * func2._evaluate(Z)
            qval = func._evaluate(E) / tau + func2._evaluate(Z)
            qvals.append(qval)

            if qval < qval_min:
                qval_min_prev = qval_min
                qval_min = qval
                inventory.move(Z_min, Z)
                # for j in range(N):
                #     if Z_min[j] < 0:
                #         Z_min[j] = 0
                # putmask(Z_min, Z_min < 0, 0)
                
            if fabs(qval - qval_prev) / (1.0 + fabs(qval_min)) < tol:
                break

            if fabs(qval_min - qval_min_prev) / (1.0 + fabs(qval_min)) < tol:
                break
                
            if qval > qval_prev:
                M += 1
                
            if M > 10:
                break

        self.Z = Z_min
        self.qval = qval_min
        self.K = K+1
        self.delta_qval = fabs(qval_min - qval_min_prev)
        self.qvals = qvals

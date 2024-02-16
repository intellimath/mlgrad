#
#
#

import numpy as np
from sklearn.linear_model import LinearRegression

class RBF:
    #
    def __init__(self, n_component, centers=None, scales=None, weights=None):
        self.n_component = n_component
        if centers is None:
            self.centers = np.zeros(n_component, 'd')
        else:
            self.centers = np.asarray(centers, 'd')
        if scales is None:
            self.scales = np.ones(n_component, 'd')
        else:
            self.scales = np.asarray(scales, 'd')
        if weights is None:
            self.weights = np.full(n_component+1, 1.0/n_component, 'd')
            self.weights[0] = 0
        else:
            self.weights = np.asarray(weights, 'd')
    #
    def evaluate(self, X):
        X = X[:,None]
        XC = X - self.centers[None,:]
        U = XC / self.scales[None,:]
        V = np.exp(-0.5*U*U) / self.scales[None,:]
        return self.weights[0] + V @ self.weights[1:]
    #
    def next_centers(self, X, Y):
        E = self.evaluate(X) - Y
        X = X[:,None]
        XC = X - self.centers[None,:]
        U = XC / self.scales[None,:]
        U_min = np.min(U, axis=1)
        V = (np.exp(-0.5*U) - np.exp(-0.5*U_min[:,None])) * E[:,None]
        s = np.abs(np.sum(V, axis=0))
        if any(abs(s) < 1.0e-9):
            return self.centers
        centers = np.abs(np.sum(V * X, axis=0)) / s
        return centers
    #
    def next_scales(self, X, Y):
        E = self.evaluate(X) - Y
        X = X[:,None]
        XC = X - self.centers[None,:]
        U = XC / self.scales[None,:]
        U = U*U
        U_min = np.min(U, axis=1)
        V = (np.exp(-0.5*U) - np.exp(-0.5*U_min[:,None])) * E[:,None]
        s = np.abs(np.sum(V, axis=0))
        if any(abs(s) < 1.0e-9):
            return self.scales
        scales = np.abs(np.sum(V * XC*XC, axis=0)) / s
        return np.sqrt(scales)
    #
    def next_weights(self, X, Y):
        X = X[:,None]
        XC = X - self.centers[None,:]
        U = XC / self.scales[None,:]
        U = U*U
        XX = np.exp(-0.5*U) / self.scales[None,:]
        lr = LinearRegression()
        lr.fit(XX, Y)
        weights = np.empty(self.n_component+1, 'd')
        weights[0] = lr.intercept_
        weights[1:] = lr.coef_[:]
        np.putmask(weights, weights<0, 1.0e-9)
        return weights
    #
    def fit(self, X, Y, n_iter=100, tol=1.0e-6):
        E = self.evaluate(X) - Y
        lval = lval_min = np.sqrt(np.mean(E*E))
        self.lvals = [lval]
        centers_min = self.centers
        scales_min = self.centers
        for k in range(n_iter):
            lval_prev = lval
            self.centers = self.next_centers(X, Y)
            self.scales = self.next_scales(X, Y)
            # print(self.centers, self.scales)
            self.weights = self.next_weights(X, Y)
            print(k, self.centers, self.scales, self.weights)

            E = self.evaluate(X) - Y
            lval = np.sqrt(np.mean(E*E))
            self.lvals.append(lval)
            print(lval)

            if lval < lval_min:
                lval_min = lval
                centers_min = self.centers.copy()
                scales_min = self.centers.copy()
                
            if lval < tol or abs(lval - lval_prev) < tol:
                break
        
        self.centers = centers_min
        self.scales = scales_min

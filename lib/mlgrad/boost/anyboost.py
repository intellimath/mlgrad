import mlgrad.models as models
import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.cls as cls
import mlgrad.inventory as inventory

import numpy as np

from scipy.optimize import minimize_scalar

def sigmoidal_factory(n):
    return models.SigmaNeuronModel(funcs.Sigmoidal(7.0), n)

class AnyBoostClassification:
    #
    def __init__(self, func=funcs.Exp(-1.0), model_factory=sigmoidal_factory, lossfunc=loss.NegMargin(), shrink=1.0, n_iter=100):
        self.func = func
        self.model_factory = model_factory
        self.lossfunc = lossfunc
        self.shrink = shrink
        self.n_iter = n_iter
    #
    def weak_learn(self, X, Y, **kw):
        mod = self.model_factory(X.shape[1])
        cls.classification_as_regr(X, Y, mod, self.lossfunc, weights=self.weights, **kw)
        # print(np.asarray(mod.param))
        return mod
    #
    def evaluate_alpha(self, mod, X, Y):
        self.M_vals = Y * self.H.evaluate(X)
        self.m_vals = Y * mod.evaluate(X)
        #
        def _func_(alpha):
            return self.func.evaluate_sum(self.M_vals + alpha * self.m_vals)
        #
        res = minimize_scalar(_func_, (-1., 1.))
        if not res.success:
            raise RuntimeError(f"K={self.K}: {res.message}")
        #
        return res.x
    #
    def evaluate_weights(self):
        self.weights = -self.func.derivative_array(self.M_vals)
    #
    def fit_step(self, X, Y, **kw):
        h = self.weak_learn(X, Y, **kw)
        alpha = self.evaluate_alpha(h, X, Y)
        if alpha <= 0:
            print(f"Warning: K={self.K} alpha={alpha:.6f}")
        else:
            self.H.add(h, self.shrink * alpha)
            #
            # A = self.H.weights.asarray()
            # A /= abs(A).sum()
            # del A
            #
            self.evaluate_weights()
    #
    def fit(self, X, Y, **kw):
        self.H = models.LinearFuncModel()
        self.M = np.ones(len(X), "d")
        self.weights = np.ones(len(X), "d")

        self.K = 1
        for K in range(self.n_iter):
            self.fit_step(X, Y, **kw)
            self.K += 1

        A = self.H.weights.asarray()
        A /= abs(A).sum()
        del A

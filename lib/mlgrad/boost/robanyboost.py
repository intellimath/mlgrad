import mlgrad.models as models
import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.af as af
import mlgrad.regr as regr
import mlgrad.inventory as inventory

from .anyboost import AnyBoostClassification

import numpy as np
import sys

from scipy.optimize import minimize_scalar

def sigmoidal_factory(n, scale=1.0):
    mod = models.SigmaNeuronModel(funcs.Sigmoidal(scale), n)
    mod.init_param(random=1)
    return mod

class RobustAnyBoostClassification(AnyBoostClassification):
    #
    def __init__(self,
                 aggfunc=af.averaging_function("AM"),
                 func=funcs.Exp(-1.0),
                 model_factory=sigmoidal_factory,
                 lossfunc=loss.NegMargin(),
                 normalizer=None,
                 min_weak_learn_score=0,
                 callback=None, shrink_model=False,
                 shrink=1.0, n_classifier=100, n_failures=10):
        #
        self.aggfunc = aggfunc
        self.func = func
        self.model_factory = model_factory
        self.lossfunc = lossfunc
        self.normalizer=normalizer,
        self.shrink = shrink
        self.callback=callback
        self.n_classifier = n_classifier
        self.n_failures = n_failures
        self.min_weak_learn_score = min_weak_learn_score
        # self.min_wc_accuracy = min_wc_accuracy
        self.shrink_model = shrink_model
    #
    def weak_learn(self, X, Y, **kw):
        weak_model = self.model_factory(X.shape[1])
        weak_learner = regr.regression(X, Y, weak_model, self.lossfunc, weights=self.weights, **kw)
        return weak_learner
    #
    def evaluate_alpha(self):
        #
        M_vals = self.M_vals
        m_vals = self.m_vals
        shrink = self.shrink
        aggfunc_weights = self.aggfunc_weights
        #
        def _func_(alpha):
            return aggfunc_weights @ self.func.evaluate_array(M_vals + alpha * m_vals) + 0.5*shrink * alpha*alpha
        #
        res = minimize_scalar(_func_, (0, 1.), method="Brent")
        if not res.success:
            raise RuntimeError(res.message)
        #
        return res.x
    #
    def evaluate_weights(self):
        V = self.func.evaluate_array(self.M_vals)
        self.lval = self.aggfunc.evaluate(V)
        self.lvals.append(self.lval)

        # self.aggfunc_weights = self.aggfunc.derivative_div(V)
        self.aggfunc_weights = self.aggfunc.weights(V)
        weights = self.aggfunc_weights * -self.func.derivative_array(self.M_vals)
        inventory.normalize(weights)
        return weights
    #
    def fit_init(self, X, Y):
        N = len(X)
        self.H = models.LinearFuncModel()
        self.M_vals = np.zeros(N, "d")
        self.weights = np.ones(N, "d") / N
        self.aggfunc_weights = np.ones(N, "d") / N
    #
    # def fit_step(self, X, Y, **kw):
    #     weak_learner = self.weak_learn(X, Y, **kw)
    #     weak_model = weak_learner.risk.model

    #     self.m_vals = Y * weak_model.evaluate(X)

    #     if self.weights @ self.m_vals <= 0:
    #         return False

    #     alpha = self.evaluate_alpha()
    #     self.H.add(weak_model, alpha)

    #     self.M_vals = Y * self.H.evaluate(X)
    #     self.weights = self.evaluate_weights()

    #     return True
    # #
    # def fit(self, X, Y, **kw):
    #     N = len(X)
    #     self.H = models.LinearFuncModel()
    #     self.M_vals = np.zeros(N, "d")
    #     self.weights = np.ones(N, "d") / N
    #     self.aggfunc_weights = np.ones(N, "d") / N
    #     # self.aggfunc_weights2 = np.ones(N, "d")

    #     self.lval_min = sys.float_info.max
    #     self.H_min = self.H.copy()

    #     self.lvals = []
    #     # self.wl_lvals = []

    #     n_classifier = self.n_classifier
    #     n_failures   = self.n_failures

    #     K = 1
    #     m = 0
    #     while K <= n_classifier:
    #         if m > n_failures:
    #             print(f"WARNING: Failed to complete fit step {m} times (K={K})")
    #             break
    #         if not self.fit_step(X, Y, **kw):
    #             m += 1
    #             continue
    #         K += 1
    #         m = 0

    #     self.H = self.H_min
    #     self.H_min = None

    #     A = self.H.weights.asarray()
    #     A /= abs(A).sum()
    #     del A

    #     self.K = K


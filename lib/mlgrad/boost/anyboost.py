import mlgrad.models as models
import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.cls as cls
import mlgrad.inventory as inventory

import numpy as np

from scipy.optimize import minimize_scalar

def sigmoidal_factory(n, scale=5.0):
    return models.SigmaNeuronModel(funcs.Sigmoidal(scale), n)

class AnyBoostClassification:
    #
    def __init__(self,
                 func=funcs.Exp(-1.0),
                 model_factory=sigmoidal_factory,
                 lossfunc=loss.NegMargin(),
                 min_wl_score=0,
                 shrink=1.0, n_classifier=100, n_failures=30):
        #
        self.func = func
        self.model_factory = model_factory
        self.lossfunc = lossfunc
        self.shrink = shrink
        self.n_classifier = n_classifier
        self.n_failures = n_failures
        self.min_wl_score = min_wl_score
        # self.min_wc_accuracy = min_wc_accuracy
    #
    def weak_learn(self, X, Y, **kw):
        weak_model = self.model_factory(X.shape[1])
        weak_learner = cls.classification_as_regr(X, Y, weak_model, self.lossfunc, weights=self.weights, **kw)
        # print(np.asarray(weak_model.param))
        return weak_learner
    #
    def evaluate_alpha(self, h, X, Y):
        #
        m_vals = Y * h.evaluate(X)
        #
        def _func_(alpha):
            return self.func.evaluate_sum(self.M_vals + alpha * m_vals)
        #
        res = minimize_scalar(_func_, (-1., 1.))
        if not res.success:
            raise RuntimeError(f"K={self.K}: {res.message}")
        #
        return res.x
    #
    def evaluate_weights(self):
        self.weights = -self.func.derivative_array(self.M_vals)
        self.weights /= self.weights.sum()
    #
    def fit_step(self, X, Y, **kw):
        weak_learner = self.weak_learn(X, Y, **kw)
        wl_model = weak_learner.risk.model

        self.M_vals = Y * self.H.evaluate(X)

        alpha = self.evaluate_alpha(wl_model, X, Y)
        if alpha <= 0:
            return False
        else:
            wl_lval = weak_learner.risk.evaluate()
            wl_lval = abs(wl_lval)
            if wl_lval <= self.min_wl_score:
                return None

            self.wl_lvals.append(wl_lval)

            self.H.add(wl_model, self.shrink * alpha)
            self.evaluate_weights()

            self.lvals.append(self.func.evaluate_sum(self.M_vals))

            return True
    #
    def fit(self, X, Y, **kw):
        self.H = models.LinearFuncModel()
        self.M = np.zeros(len(X), "d")
        self.weights = np.ones(len(X), "d")

        self.lvals = []
        self.wl_lvals = []

        n_classifier = self.n_classifier
        n_failures   = self.n_failures

        K = 1
        m = 0
        while K <= n_classifier:
            if m > n_failures:
                print(f"WARNING: Failed to complete fit step {m} times (K={K})")
                break
            if not self.fit_step(X, Y, **kw):
                m += 1
                continue
            K += 1
            m = 0

        A = self.H.weights.asarray()
        A /= abs(A).sum()
        del A
        self.K = K

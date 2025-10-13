import mlgrad.models as models
import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.cls as cls
import mlgrad.inventory as inventory

import numpy as np

from scipy.optimize import minimize_scalar

def sigmoidal_factory(n, scale=10.0):
    return models.SigmaNeuronModel(funcs.Sigmoidal(scale), n)

class AnyBoostClassification:
    #
    def __init__(self,
                 func=funcs.Exp(-1.0),
                 model_factory=sigmoidal_factory,
                 lossfunc=loss.NegMargin(),
                 H=None,
                 n_retry=3,
                 shrink=0.1, n_classifier=100, n_failures=30):
        #
        self.func = func
        self.model_factory = model_factory
        self.lossfunc = lossfunc
        self.shrink = shrink
        self.n_retry = n_retry
        self.n_classifier = n_classifier
        self.n_failures = n_failures
        self.H = None
        if H is not None:
            self.H = H
    #
    def weak_learn(self, X, Y, **kw):
        weak_model = self.model_factory(X.shape[1])
        weak_learner = cls.classification_erm(X, Y, weak_model, self.lossfunc,
                                              weights=self.weights, **kw)
        return weak_learner
    #
    def weak_margins(self, X, Y):
        return np.fromiter(((Y * mod.evaluate(X)).mean() for mod in self.H.models), "d", len(self.H.models))
    #
    def evaluate_alpha(self):
        #
        m_vals = self.m_vals
        M_vals = self.M_vals
        N = len(M_vals)
        shrink = self.shrink
        #
        def _func_(alpha):
            return self.func.evaluate_sum(M_vals + alpha * m_vals) / N + shrink * alpha*alpha
        #
        res = minimize_scalar(_func_, (0, 1.))
        if not res.success:
            raise RuntimeError(res.message)
        #
        return res.x
    #
    def evaluate_weights(self):
        weights = -self.func.derivative_array(self.M_vals)
        inventory.normalize(weights)
        return weights
    #
    def fit_step(self, X, Y, **kw):
        N = len(Y)
        weak_learner = self.weak_learn(X, Y, **kw)
        weak_model = weak_learner.risk.model
        self.m_vals = Y * weak_model.evaluate(X)

        if (self.weights @ self.m_vals) < 0:
            return False

        alpha = self.evaluate_alpha()
        self.H.add(weak_model, alpha)

        self.M_vals = Y * self.H.evaluate(X)
        lval = self.func.evaluate_sum(self.M_vals) / N
        self.lvals.append(lval)

        self.weights = self.evaluate_weights()

        return True
    #
    def fit(self, X, Y, **kw):
        N = len(X)
        if self.H is None:
            self.H = models.LinearFuncModel()
        self.M_vals = np.zeros(N, "d")
        self.weights = np.ones(N, "d") / N

        self.lvals = []

        n_classifier = self.n_classifier
        n_failures   = self.n_failures

        K = len(self.H.models)
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
        inventory.normalize(A)
        del A
        self.K = K

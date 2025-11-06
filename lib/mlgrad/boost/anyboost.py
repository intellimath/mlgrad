import mlgrad.models as models
import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.regr as regr
import mlgrad.inventory as inventory

import numpy as np

from scipy.optimize import minimize_scalar

def sigmoidal_factory(n, scale=10.0):
    return models.SigmaNeuronModel(funcs.Sigmoidal(scale), n)

class AnyBoostClassification:
    #
    def __init__(self,
                 func=funcs.SoftHinge_Exp(1.0),
                 model_factory=sigmoidal_factory,
                 lossfunc=loss.NegMargin(),
                 normalizer=None,
                 H=None,
                 alpha_method="newton",
                 callback=None,
                 n_retry=3, shrink=0.1, n_classifier=100, n_failures=10):
        #
        self.func = func
        self.model_factory = model_factory
        self.lossfunc = lossfunc
        self.evaluate_alpha = {
            "linesearch": self._evaluate_alpha_linesearch,
            "newton": self._evaluate_alpha_linesearch}[alpha_method]
        self.callback=callback
        self.normalizer=normalizer
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
        weak_learner = regr.regression(
                X, Y, weak_model, self.lossfunc,
                weights=self.weights, normalizer=self.normalizer, **kw)
        return weak_learner
    #
    def weak_margins(self, X, Y):
        return np.fromiter(((Y * mod.evaluate(X)).mean() for mod in self.H.models), "d", len(self.H.models))
    #
    def _evaluate_alpha_linesearch(self):
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
    def _evaluate_alpha_newton(self):
        #
        m_vals = self.m_vals
        mval1 = self.func.derivative_array(m_vals).sum()
        mval2 = self.func.derivative2_array(m_vals).sum()

        return self.shrink * mval1 / mval2
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

        wlval = self.weights @ self.m_vals
        if wlval <= 0:
            return False

        self.wlvals.append(wlval)

        alpha = self.evaluate_alpha()
        self.H.add(weak_model, alpha)

        self.M_vals = Y * self.H.evaluate(X)
        lval = self.func.evaluate_sum(self.M_vals) / N
        self.lvals.append(lval)

        # A = self.H.weights.asarray()
        # inventory.normalize(A)
        # del A

        self.weights = self.evaluate_weights()

        if self.callback:
            self.callback(self)

        return True
    #
    def classifier(self):
        return models.SimpleComposition(funcs.Sign(), self.H)
    #
    def fit(self, X, Y, **kw):
        N = len(X)
        if self.H is None:
            self.H = models.LinearFuncModel()
        self.M_vals = np.zeros(N, "d")
        self.weights = np.ones(N, "d") / N

        self.lvals = []
        self.wlvals = []

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

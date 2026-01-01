import mlgrad.models as models
import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.regr as regr
import mlgrad.inventory as inventory

import numpy as np
import sys

from scipy.optimize import minimize_scalar, line_search

def sigmoidal_factory(n, scale=1.0):
    mod = models.SigmaNeuronModel(funcs.Sigmoidal(scale), n)
    mod.init_param(random=1)
    return mod

class AnyBoostClassification:
    #
    def __init__(self,
                 func=funcs.SoftHinge_Exp(1.0),
                 model_factory=sigmoidal_factory,
                 lossfunc=loss.NegMargin(),
                 normalizer=None,
                 # H=None,
                 alpha_method="linesearch",
                 callback=None, shrink_model=False,
                 n_retry=3, shrink=1.0, n_classifier=100, n_failures=10):
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
        self.shrink_model = shrink_model
        self.n_retry = n_retry
        self.n_classifier = n_classifier
        self.n_failures = n_failures
        self.H = None
        # if H is not None:
        #     self.H = H
    #
    def weak_learn(self, X, Y, **kw):
        weak_model = self.model_factory(X.shape[1])
        weak_learner = regr.regression(
                X, Y, weak_model, self.lossfunc,
                weights=self.weights, normalizer=self.normalizer, **kw)
        return weak_learner
    #
    # def weak_margins(self, X, Y):
    #     return np.fromiter(((Y * mod.evaluate(X)).mean() for mod in self.H.models), "d", len(self.H.models))
    # #
    def _evaluate_alpha_linesearch(self):
        #
        m_vals = self.m_vals
        M_vals = self.M_vals
        shrink = self.shrink
        #
        def _func_(alpha):
            return self.func.evaluate_array(M_vals + alpha * m_vals).mean() + 0.5*shrink * alpha*alpha
        #
        res = minimize_scalar(_func_, (0, 1.), method="Brent")
        if not res.success:
            raise RuntimeError(res.message)
        #
        return res.x
    #
    def _evaluate_alpha_newton(self):
        m_vals = self.m_vals
        M_vals = self.M_vals
        v1 = self.func.derivative_array(M_vals) @ m_vals
        v2 = self.func.derivative2_array(M_vals) @ (m_vals * m_vals)

        return self.shrink * v1 / v2
    #
    def evaluate_weights(self):
        self.lval = self.func.evaluate_array(self.M_vals).mean()
        self.lvals.append(self.lval)

        weights = -self.func.derivative_array(self.M_vals)
        inventory.normalize(weights)
        return weights
    #
    def evaluate_weights_ext(self, alpha):
        weights = -self.func.derivative_array(self.M_vals)
        weights -= 0.5 * alpha * self.func.derivative2_array(self.M_vals * self.m_vals)
        inventory.normalize(weights)
        return weights
    #
    def fit_step(self, X, Y, **kw):
        weak_learner = self.weak_learn(X, Y, **kw)
        weak_model = weak_learner.risk.model
        self.m_vals = Y * weak_model.evaluate(X)

        if self.weights @ self.m_vals <= 0:
            return False

        alpha = self.evaluate_alpha()

        self.H.add(weak_model, alpha)

        self.M_vals = Y * self.H.evaluate(X)
        self.weights = self.evaluate_weights()

        errval = (np.sign(self.H.evaluate(X)) != Y).astype("d").mean()
        self.errvals.append(errval)

        # if self.lval < self.lval_min:
        #     self.lval_min = self.lval
        #     self.H_min = self.H.copy()

        if errval < self.errval_min:
            self.errval_min = errval
            self.H_min = self.H.copy()

        if self.callback:
            self.callback(self)

        return True
    #
    def classifier(self):
        return models.SimpleComposition(funcs.Sign(), self.H)
    #
    def fit_init(self, X, Y):
        N = len(X)
        if self.H is None:
            self.H = models.LinearFuncModel()
        self.M_vals = np.zeros(N, "d")
        self.weights = np.ones(N, "d") / N
    #
    def fit(self, X, Y, **kw):
        self.fit_init(X, Y)

        self.lvals = []
        self.errvals = []

        self.errval_min = sys.float_info.max
        self.lval_min = sys.float_info.max
        # self.m_min = 0

        n_classifier = self.n_classifier
        n_failures   = self.n_failures

        self.K = len(self.H.models)
        m = 0
        while self.K <= n_classifier:
            if m > n_failures:
                print(f"WARNING: Failed to complete fit step {m} times (K={self.K})")
                break

            if not self.fit_step(X, Y, **kw):
                m += 1
                continue

            self.K += 1
            m = 0

        self.H = self.H_min
        self.H_min = None

        # if self.shrink_model:
        #     H = models.LinearFuncModel()
        #     for i in range(self.m_min):
        #         H.add(self.H.models[i], self.H.weights[i])
        #     self.H = H

        A = self.H.weights.asarray()
        inventory.normalize(A)
        del A



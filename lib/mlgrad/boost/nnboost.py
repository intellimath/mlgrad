from mlgrad.cls import classification_erm
from mlgrad.models import models

from scipy.optimize import minimize_scalar

def make_nnmodel(n_classifier, clsmodel):
    pass

class BoostModel:
    #
    def __init__(self, n_classifier, weak_model_creator):
        self.n_classifier = n_classifier
        self.linear_model = models.LinearModel(n_classifier)
        self.linear_model.param = np.ones(n_classifier, "d") / n_classifier
        self.weak_models = []
        for i in range(n_classifier):
            weak_model = weak_model_creater()
            weak_model.init_param()
            self.weak_models.append(weak_model)
    #
    def evaluate_weak_models(self, X):
        N = X.shape[0]
        U = np.empty((N, self.n_classifier), "d")
        for i in range(self.n_classifier):
            mod = self.weak_models[i]
            U[:,i] = mod.evaluate(X)
        return U
    #
    def evaluate(self, X):
        U = self.evaluate_weak_models(X)
        Y = self.linear_model.evaluate(U)
        return Y
    #
    def classifier(self, X):
        return np.sign(self.evaluate(X))


class AnyBoostRepeated:
    #
    def __init__(self, boost_model, loss_func, regularizer_creator=None, tau=0, shrink=0.1, h=0.1, n_iter=10):
        self.boost_model = boost_model
        self.n_classifier = boost_model.n_classifier
        self.neg_loss_func = loss.NegMarginLoss()
        self.loss_func = loss_func
        self.alpha = boost_model.param
        self.h = h
        #
        if regularizer_creator:
            for i in range(n_classifier):
                reg = regularizer_creator()
                weak_models[i].use_regularizer(regn, self.tau)
        self.tau = tau
        self.n_iter = n_iter
    #
    def fit_step(self, X, Y):
        N = len(X)
        U = self.U
        weak_models = self.boost_model.weak_models
        shrink = self.shrink
        for j in range(self.n_classifier):
            V = -self.loss_func.derivative_all(SU_j, Y)
            weak_model = weak_models[j]
            classification_erm(X, Y,
                               weak_model,
                               self.neg_loss_func,
                               weights=V,
                               h=self.h)

            U_j = U[:,j] = model_j.evaluate(X)

            alpha_j = self.alpha[j]
            self.alpha[j] = 0
            SU_j = U @ self.alpha
            self.alpha[j] = alpha_j

            def func_alpha(alpha):
                return self.loss_func.evaluate_all(SU_j + alpha * U_j, Y).sum()

            res = minimize_scalar(func_alpha, (0, 1.))
            alpha = res.x
            self.alpha[j] = shrink * alpha

        alpha[:] = alpha / abs(alpha).sum()
    #
    def fit(X, Y):
        self.U = np.zeros(N, "d")

        K = 0
        self.fit_step(X, Y)
        lval = lval_min = self.loss_func.evaluate_all(self.boost_model.evaluate(X), Y).mean()
        linear_model_param = self.boost_model.linear_model.param.copy()
        weak_models_params = [mod.param.copy() for mod in self.boost_model.weak_models]

        lvals = [lval]

        K += 1
        while K < self.n_iter:
            self.fit_step(X, Y)
            lval = self.loss_func.evaluate_all(self.boost_model.evaluate(X), Y).mean()
            lvals.append(lval)

            if lval < lval_min:
                lval_min = lval
                linear_model_param = self.boost_model.linear_model.param.copy()
                weak_models_params = [mod.param.copy() for mod in self.boost_model.weak_models]
            K += 1

        self.boost_model.linear_model.param[:] = linear_model_param
        for mod, param in zip(self.boost_model.weak_models, weak_models_params):
            mod.param[:] = param
        self.K = K+1
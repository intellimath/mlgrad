#
# Gradient Boosting Regression 
#

from mlgrad.risk import ERisk, ERiskGB
from mlgrad.loss import MarginLoss
from mlgrad.func import Hinge, HSquare, Square
from mlgrad.model import LinearFuncModel

from mlgrad import erm_fg, erisk
from mlgrad.af import averaging_function

import numpy as np

np_dot = np.dot
np_zeros = np.zeros
np_double = np.double

class GradientBoostingClassification:
    
    def __init__(self, complex_model, new_model, loss_func=None, 
                 agg=None, h=0.001, n_iter=100, n_iter2=22, tol=1.0e-9):
        self.complex_model = complex_model
        self.new_model = new_model
        if loss_func is None:
            self.loss_func = MarginLoss(Hinge())
        else:
            self.loss_func = loss_func
        self.h = h
        self.n_iter = n_iter
        self.n_iter2 = n_iter2
        self.tol = tol
    #
    def find_alpha(self, risk):
        HY = risk.H.base * risk.Y.base
        YY = risk.model.evaluate_all(risk.X) * risk.Y.base
        M = HY + risk.alpha * YY
        W = self.loss_func.func.derivative_div_array(M)

        s = np.sum(W)
        if s != 0:
            W /= s
        
        WYY = W * YY
        risk.alpha = (WYY @ self.loss_func.func.value_array(HY)) / (WYY @ YY)
    #
    def find_param(self, risk):
        # Y = risk.model.evaluate_all(risk.X)
        # HY = risk.H.base * risk.Y.base
        # YY = Y * risk.Y.base
        M = (risk.H.base + risk.alpha * risk.model.evaluate_all(risk.X)) * risk.Y.base
        W = self.loss_func.func.derivative_div_array(M)

        s = np.sum(W)
        if s != 0:
            W /= s

        risk.use_weights(W)
        return erm_fg(risk, h=self.h, tol=self.tol)
    #
    def find_param_alpha(self, risk):
        lval_min = lval = risk.evaluate()
        param_min = risk.model.param.copy()
        alpha_min = 1.0
        finish = 0

        for j in range(self.n_iter2):
            lval_prev = lval

            self.find_param(risk)

            self.find_alpha(risk)

            lval = risk.evaluate()

            if j > 0 and abs(lval - lval_min) / (1 + abs(lval_min)) < self.tol:
                finish = 1
            
            if lval < lval_min:
                param_min = risk.model.param.copy()
                alpha_min = risk.alpha
                lval_min = lval
                
            if finish:
                break

        risk.model.param[:] = param_min
        risk.alpha = alpha_min
    #
    def fit(self, X, Y):        
        n = X.shape[1]

        self.lvals = []

        for k in range(self.n_iter):
            # print(k)
            mod = self.new_model(n)
            # risk = ERiskGB(X, Y, mod, self.loss_func)
            risk = ERiskGB(X, Y, mod, MarginLoss(HSquare()))
            risk.H[:] = self.complex_model.evaluate_all(X)
            
            self.find_param_alpha(risk)

            lval = np.mean(self.loss_func.evaluate_all(self.complex_model.evaluate_all(X), Y))
            # lval = trisk.evaluate()
            self.lvals.append(lval)

            self.complex_model.add(mod, risk.alpha)            


class MGradientBoostingClassification:
    
    def __init__(self, complex_model, new_model, loss_func=None, agg=None, 
                 h=0.001, n_iter=100, n_iter2=11, tol=1.0e-9):
        self.complex_model = complex_model
        self.new_model = new_model
        if loss_func is None:
            self.loss_func = MarginLoss(Hinge())
        else:
            self.loss_func = loss_func
        if agg is None:
            self.agg = averaging_function('WM')
        else:
            self.agg = agg
        self.h = h
        self.n_iter = n_iter
        self.n_iter2 = n_iter2
        self.tol = tol
    #
    def find_alpha(self, risk, W):
        Yh = risk.evaluate_models() * risk.Y.base
        R = W * Yh
        E = 1 -  Yh
        np.putmask(E, E < 0, 0)
        alpha = (R @ E) / (R @ Yh)
        risk.alpha = alpha
    #
    def find_param(self, risk, W):
        risk.use_weights(W)
        alg = erm_fg(risk, h=self.h, tol=self.tol)
        return alg
    #
    def find_param_alpha(self, risk):
        alpha_min = 1.0
        n = risk.batch.size
        finish = 0

        risk.evaluate_models()
        L = risk.evaluate_losses()
        self.agg.fit(L)
        W = self.agg.gradient(L)
        D = risk.evaluate_losses_derivative_div()
        W *= -D
        W /= np.sum(W)
        lval = lval_min = self.agg.u
        
        param_min = risk.model.param.copy()
        alpha_min = 1

        for j in range(self.n_iter2):
            lval_prev = lval
            
            self.find_param(risk, W)

            self.find_alpha(risk, W)

            risk.evaluate_models()
            L = risk.evaluate_losses()
            self.agg.fit(L)
            W = self.agg.gradient(L)
            D = risk.evaluate_losses_derivative_div()
            W *= -D
            W /= np.sum(W)
            lval = self.agg.u
            # lval = risk.evaluate()

            if j > 0 and abs(lval - lval_min) / (1 + abs(lval_min)) < self.tol:
                finish = 1
            
            if lval < lval_min:
                param_min = risk.model.param.copy()
                alpha_min = risk.alpha
                lval_min = lval

            if finish:
                break

        risk.model.param[:] = param_min
        risk.alpha = alpha_min
    #
    def fit(self, X, Y):
        n = X.shape[1]
        self.lvals = []
        for k in range(self.n_iter):
            # print(k)
            mod = self.new_model(n)
            risk = ERiskGB(X, Y, mod, MarginLoss(HSquare()))
            risk.H[:] = self.complex_model.evaluate_all(X)
    
            self.find_param_alpha(risk)

            L = risk.evaluate_losses()
            self.agg.fit(L)
            self.lvals.append(self.agg.u)

            self.complex_model.add(mod, risk.alpha)

def gb_fit(X, Y, new_model, loss_func=None, 
           h=0.001, n_iter=100, n_iter2=11, tol=1.0e-9):
    lfm = LinearFuncModel()
    gb = GradientBoostingClassification(lfm, new_model, loss_func, 
                          h=h, n_iter=n_iter, n_iter2=n_iter2, tol=tol)
    gb.fit(X, Y)
    return gb

def gb_fit_agg(X, Y, new_model, loss_func=None, aggname='WM', 
               alpha=0.5, h=0.001, n_iter=100, n_iter2=10, tol=1.0e-9):
    lfm = LinearFuncModel()
    agg = averaging_function(aggname, alpha=alpha)
    gb = MGradientBoostingClassification(lfm, new_model, loss_func, agg, 
                           h=h, n_iter=n_iter, n_iter2=n_iter2, tol=tol)
    gb.fit(X, Y)
    return gb
        
#
#
#

from mlgrad.avragg import PenaltyAverage, PenaltyScale, Average_FG, ArithMean, ParameterizedAverage
from mlgrad.gd import FG, FG_RUD, SGD
from mlgrad.risks import ED, MRisk, ERisk, ERisk2, ERisk22, SimpleFunctional
from mlgrad.irgd import IRGD
from mlgrad.averager import ArraySave, ArrayMOM, ArrayRMSProp, ArrayAMOM, ArrayAdaM1, ArrayAdaM2, ScalarAdaM1, ScalarAdaM2
import numpy as np

from mlgrad.loss import Loss, MultLoss, MultLoss2

__all__ = ['averager_it', 'average_it', 'ws_average_it', 'averager_fg', 'average_fg', 
           'fg', 'erm_fg', 'erm_fg_rud', 'sg', 'erm_sg',
           'irgd', 'erm_irgd', 'erisk', 'erisk2']


__averager_dict = {
    None : ArraySave,
    'AMom':ArrayAMOM,
    'Mom':ArrayMOM,
    'RMS':ArrayRMSProp,
    'AdaM1':ArrayAdaM1,
    'AdaM2':ArrayAdaM2,
}

__averager_scalar_dict = {
    'AdaM1':ScalarAdaM1,
    'AdaM2':ScalarAdaM2,
}

def drisk(X, distfunc, regnorm=None, weights=None, tau=0.001):
    er = ED(X, distfunc)
    if weights is not None:
        er.use_weights(weights)
    return er

def sfunc(func):
    f = SimpleFunctional(func)
    return f

def erisk(X, Y, mod, loss_func, regnorm=None, weights=None, tau=0.001, batch=None):
    if isinstance(loss_func, MultLoss):
        er = ERisk21(X, Y, mod, loss_func, regnorm=regnorm, tau=tau, batch=batch)
    elif isinstance(loss_func, MultLoss2):
        er = ERisk22(X, Y, mod, loss_func, regnorm=regnorm, tau=tau, batch=batch)
    else:
        er = ERisk(X, Y, mod, loss_func, regnorm=regnorm, tau=tau, batch=batch)
    if weights is not None:
        er.use_weights(weights)
    return er

def mrisk(X, Y, mod, loss_func, avg, regnorm=None, weights=None, tau=0.001, batch=None):
    mr = MRisk(X, Y, mod, loss_func, avg, regnorm=regnorm, tau=tau, batch=batch)
    if weights is not None:
        mr.use_weights(weights)
    return mr

# def aerisk(X, Y, mod, loss_func, avr=None, regnorm=None, tau=0.001, batch=None):
#     er = AER(X, Y, mod, loss_func, avr, regnorm=regnorm, tau=tau, batch=batch)
#     return er

def erisk2(X, Y, mod, loss_func, regnorm=None, weights=None, tau=0.001, batch=None):
    er = ERisk2(X, Y, mod, loss_func, regnorm=regnorm, tau=tau, batch=batch)
    if weights is not None:
        er.use_weights(weights)
    return er

def erisk22(X, Y, mod, loss_func, regnorm=None, weights=None, tau=0.001, batch=None):
    er = ERisk22(X, Y, mod, loss_func, regnorm=regnorm, tau=tau, batch=batch)
    if weights is not None:
        er.use_weights(weights)
    return er

def averager_it(func, tol=1.0e-6, n_iter=1000):
    penalty = PenaltyAverage(func)
    alg = Average_Iterative(penalty, tol=tol, n_iter=n_iter)
    return alg

def averager_fg(func, h=0.01, tol=1.0e-6, n_iter=1000, averager=None):
    penalty = PenaltyAverage(func)
    alg = Average_FG(penalty, h=h, tol=tol, n_iter=n_iter)
    _averager = __averager_scalar_dict.get(averager, None)
    if _averager is not None:
        alg.use_gradient_averager(_averager())
    return alg

def average_it(Y, func, tol=1.0e-6, n_iter=1000, verbose=0):
    alg = averager_it(func, tol=tol, n_iter=n_iter)
    alg.fit(np.asarray(Y, 'd'))
    if verbose:
        print("K={} u={}".format(alg.K, alg.u, alg.s))
    return alg

def ws_average_it(ws_func, func, tol=1.0e-6, n_iter=1000, verbose=0):
    avr = averager_it(func, tol=tol, n_iter=n_iter)
    alg = ParametrizedAverage(ws_func, avr)
    return alg

def average_fg(Y, penalty_func, h=0.01, tol=1.0e-5, n_iter=1000, verbose=0, averager=None):
    alg = averager_fg(penalty_func, h=h, tol=tol, n_iter=n_iter, averager=averager)
    alg.fit(Y)
    if verbose:
        print("K={} u={}".format(alg.K, alg.u))
    return alg

def fg(er, h=0.001, tol=1.0e-8, n_iter=1000, averager='AdaM2', callback=None, 
       stop_condition='diffL1', normalizer=None):
    alg = FG(er, h=h, tol=tol, n_iter=n_iter, callback=callback, stop_condition=stop_condition)
    _averager = __averager_dict.get(averager, None)
    if _averager is not None:
        alg.use_gradient_averager(_averager())
    if normalizer is not None:
        alg.use_normalizer(normalizer)
    return alg

def fg_rud(er, h=0.001, tol=1.0e-8, n_iter=1000, gamma=1, averager='AdaM2', callback=None, stop_condition='diffL1', normalizer=None):
    alg = FG_RUD(er, h=h, tol=tol, n_iter=n_iter, callback=callback, stop_condition=stop_condition, gamma=gamma)
    _averager = __averager_dict.get(averager, ArrayMOM)
    if _averager is not None:
        alg.use_gradient_averager(_averager())
    return alg

def erm_fg(er, h=0.001, tol=1.0e-8, n_iter=1000, averager='AdaM2', callback=None, 
           stop_condition='diffL1', n_restart=1, verbose=0, normalizer=None):
    K = 0
    alg = fg(er, h=h, tol=tol, n_iter=n_iter,
             averager=averager, callback=callback, stop_condition=stop_condition, normalizer=normalizer)
    if normalizer is not None:
        alg.use_normalizer(normalizer)
    for i in range(n_restart):
        alg.fit()
        K += alg.K
        if alg.completed:
            break
    alg.K = K
    if verbose:
        print("K={} param={}".format(alg.K, er.param.base))
    return alg

def erm_fg_rud(er, h=0.001, tol=1.0e-6, n_iter=1000, gamma=1, averager='AdaM2', callback=None, 
               stop_condition='diffL1', n_restart=1, verbose=0, normalizer=None):
    K = 0
    for i in range(n_restart):
        alg = fg_rud(er, h=h, tol=tol, n_iter=n_iter,
                 averager=averager, callback=callback, 
                 stop_condition=stop_condition, gamma=gamma, normalizer=normalizer)
        alg.fit()
        K += alg.K
        if alg.completed:
            break
#         if i > 0:
#             alg.learn_rate.h *= 0.5
    alg.K = K
    if verbose:
        print("K={} param={}".format(alg.K, er.param.base))
    return alg

def sg(er, h=0.001, tol=1.0e-6, n_iter=1000, 
       averager='adaM1', callback=None, stop_condition='diffL1'):
    alg = SGD(er, h=h, tol=tol, n_iter=n_iter, callback=callback, stop_condition=stop_condition)
    _averager = __averager_dict.get(averager, None)
    if _averager is not None:
        alg.use_gradient_averager(_averager())
    return alg

def erm_sg(er, h=0.001, tol=1.0e-6, n_iter=1000, 
           averager='adaM1', callback=None, stop_condition='diffL1', n_restart=1, verbose=1):
    for i in range(n_restart):
        alg = sg(er, h=h, tol=tol, n_iter=n_iter, 
                 averager=averager, callback=callback, stop_condition=stop_condition)
        alg.fit()
    if verbose:
        print(alg.K, er.param.base)
    return alg

def irgd(fg, weights, tol=1.0e-4, n_iter=100, param_averager=None, callback=None, h_anneal=0.999):
    alg = IRGD(fg, weights, tol=tol, n_iter=n_iter, callback=callback, h_anneal=h_anneal)
    if param_averager is not None:
        alg.use_param_averager(param_averager)
    return alg

def erm_irgd(fg, weights, tol=1.0e-4, n_iter=100, 
             param_averager=None, callback=None, verbose=0, h_anneal=0.999, n_restart=1):
    alg = irgd(fg, weights, tol=tol, n_iter=n_iter, 
               param_averager=param_averager, callback=callback, h_anneal=h_anneal)
    for i in range(n_restart):
        alg.fit()
        if alg.completed:
            break
    if verbose:
        print("K={} param={}".format(alg.K, fg.risk.param.base))
    return alg
    
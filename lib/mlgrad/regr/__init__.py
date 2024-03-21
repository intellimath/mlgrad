# coding: utf-8 

# The MIT License (MIT)
#
# Copyright (c) <2015-2024> <Shibzukhov Zaur, szport at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import mlgrad.loss as loss
import mlgrad.funcs as func
import mlgrad.gd as gd
import mlgrad.weights as weights
from mlgrad.utils import exclude_outliers

from mlgrad.funcs2 import SquareNorm
from mlgrad.loss import SquareErrorLoss, ErrorLoss

from mlgrad import fg, erm_fg, erm_irgd, erisk, mrisk

from mlgrad.af import averaging_function, scaling_function

__all__ = 'regression', 'm_regression', 'm_regression_irls', 'r_regression_irls', 'mr_regression_irls'

def regression(Xs, Y, mod, loss_func=None, regnorm=None, *,
               normalizer=None,
               h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации эмпирического риска. 
    Параметр `loss_func` задает функцию потерь.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    if loss_func is None:
        _loss_func = SquareErrorLoss()
    else:
        _loss_func = loss_func

    if mod.param is None:
        mod.init_param()

    er = erisk(Xs, Y, mod, _loss_func, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, 
                 verbose=verbose, n_restart=n_restart, normalizer=normalizer)
    return alg

def m_regression(Xs, Y, mod, 
                 loss_func=None, agg_func=None, regnorm=None, 
                 h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
        
    if loss_func is None:
        _loss_func = SquareErrorLoss()
    else:
        _loss_func = loss_func

    if agg_func is None:
        _agg_func = averaging_function('WM')
    else:
        _agg_func = agg_func

    if mod.param is None:
        mod.init_param()

    er = mrisk(Xs, Y, mod, loss_func, _agg_func, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression_irls(Xs, Y, mod, 
                      loss_func=None,
                      agg_func=None, regnorm=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=22, verbose=0):
    """\
    Поиск параметров модели `mod` на основе принципа минимизации усредняющей 
    агрегирующей функции потерь от ошибок. 
    Параметр `avrfunc` задает усредняющую агрегирующую функцию.
    Параметр `lossfunc` задает функцию потерь.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    if loss_func is None:
        _loss_func = SquareErrorLoss()
    else:
        _loss_func = loss_func

    if agg_func is None:
        _agg_func = averaging_function('WM')
    else:
        _agg_func = agg_func

    if mod.param is None:
        mod.init_param()

    er = erisk(Xs, Y, mod, _loss_func, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)

    wt = weights.MWeights(_agg_func, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)

    return irgd

def mr_regression_irls(Xs, Y, mod, 
                       loss_func=None,
                       agg_func=None, regnorm=None, 
                       h=0.001, tol=1.0e-9, n_iter=1000,
                       tol2=1.0e-5, n_iter2=22, tau=0.001, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от ошибок. 
    Параметр `avrfunc` задает усредняющую агрегирующую функцию.
    Параметр `lossfunc` задает функцию потерь.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    if loss_func is None:
        _loss_func = ErrorLoss(funcs.Sqrt(0.001))
    else:
        _loss_func = loss_func

    if agg_func is None:
        _agg_func = averaging_function('WM')
    else:
        _agg_func = agg_func
    
    if mod.param is None:
        mod.init_param()

    er2 = erisk(Xs, Y, mod, SquareErrorLoss(), regnorm=regnorm, tau=tau)
    alg = fg(er2, h=h, tol=tol, n_iter=n_iter)

    er = erisk(Xs, Y, mod, _loss_func, regnorm=regnorm, tau=tau)
    # wt1 = weights.MWeights(_agg_func, er, normalize=0)
    # wt2 = weights.RWeights(er, normalize=0)
    
    wt = weights.WRWeights(_agg_func, er, normalize=1)

    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
        
    return irgd

def r_regression_irls(Xs, Y, mod, rho_func=None, regnorm=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=22, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи классического методо R-регрессии. 
    Параметр `rhofunc` задает функцию от ошибки.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    if rho_func is None:
        _rho_func = funcs.Square()
    else:
        _rho_func = rho_func
    loss_func = ErrorLoss(_rho_func)

    if mod.param is None:
        mod.init_param()

    er = erisk(Xs, Y, mod, SquareErrorLoss(), regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, n_iter=n_iter, tol=tol)

    er2 = erisk(Xs, Y, mod, loss_func, regnorm=regnorm, tau=tau)
    wt = weights.RWeights(er2)

    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
    return irgd


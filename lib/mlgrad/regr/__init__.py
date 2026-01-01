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
import mlgrad.funcs as funcs
import mlgrad.gd as gd
import mlgrad.weights as weights
from mlgrad.utils import exclude_outliers

from mlgrad.funcs2 import SquareNorm
from mlgrad.loss import SquareErrorLoss, ErrorLoss

from mlgrad import fg, erm_fg, erm_irgd, erisk

from mlgrad.af import averaging_function, scaling_function

__all__ = 'regression', 'm_regression', 'm_regression_irls', 'r_regression_irls', 'mr_regression_irls'

def regression(Xs, Y, mod,
               loss_func=SquareErrorLoss(),
               regnorm=None,
               *,
               weights=None,
               normalizer=None,
               h=0.001, tol=1.0e-6, n_iter=1000, tau=0.0, verbose=0,
               n_restart=1, init_param=1):
    """\
    Поиск параметров модели для решения задачи регрессии на основе принципа минимизации эмпирического риска.
    Параметры:
        Xs: входной 2-мерный массив
        Y: массив ожидаем
        mod: модель зависимости, параметры которой ищутся
        regnorm: регуляризатор
        weights: объект для вычисления весов
        normalizer: нормализатор параметров модели
        loss_func: функция потерь
    """
    if mod.param is None or init_param:
        mod.init_param()
    if regnorm is not None or tau != 0:
        mod.use_regularizer(regnorm, tau)

    er = erisk(Xs, Y, mod, loss_func)
    if weights is not None:
        er.use_weights(weights)

    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter,
                 verbose=verbose, n_restart=n_restart, normalizer=normalizer)
    return alg

def m_regression(Xs, Y, mod,
                 loss_func=SquareErrorLoss(),
                 agg_func=averaging_function('WM'),
                 regnorm=None,
                 h=0.001, tol=1.0e-9, n_iter=1000, tau=0.0, verbose=0,
                 n_restart=1, init_param=1):

    if mod.param is None:
        mod.init_param()
    if regnorm is not None:
        mod.use_regularizer(regnorm, tau)

    er = erisk(Xs, Y, mod, loss_func, avg=agg_func)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose,
                 n_restart=n_restart)
    return alg

def m_regression_irls(Xs, Y, mod,
                      loss_func=SquareErrorLoss(),
                      agg_func=averaging_function('WM'),
                      regnorm=None,
                      normalizer=None,
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.0, tol2=1.0e-5, n_iter2=22, 
                      verbose=0, init_param=1):
    """\
    Поиск параметров модели `mod` на основе принципа минимизации усредняющей 
    агрегирующей функции потерь от ошибок. 
    Параметр `avrfunc` задает усредняющую агрегирующую функцию.
    Параметр `lossfunc` задает функцию потерь.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    if mod.param is None or init_param:
        mod.init_param()
    if regnorm is not None:
        mod.use_regularizer(regnorm, tau)

    er = erisk(Xs, Y, mod, loss_func)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)
    if normalizer is not None:
        alg.use_normalizer(normalizer)

    irgd = erm_irgd(alg, agg_func, n_iter=n_iter2, tol=tol2, verbose=verbose)

    return irgd

def mr_regression_irls(Xs, Y, mod,
                       loss_func=ErrorLoss(funcs.SoftAbs_Sqrt(0.001)),
                       agg_func=averaging_function('WM'),
                       regnorm=None,
                       normalizer=None,
                       h=0.001, tol=1.0e-9, n_iter=1000,
                       tol2=1.0e-5, n_iter2=22, tau=0.0, 
                       verbose=0, init_param=1):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от ошибок. 
    Параметр `avrfunc` задает усредняющую агрегирующую функцию.
    Параметр `lossfunc` задает функцию потерь.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    if mod.param is None or init_param:
        mod.init_param()
    if regnorm is not None:
        mod.use_regularizer(regnorm, tau)

    er2 = erisk(Xs, Y, mod, SquareErrorLoss())
    alg = fg(er2, h=h, tol=tol, n_iter=n_iter)
    if normalizer is not None:
        alg.use_normalizer(normalizer)

    # er = erisk(Xs, Y, mod, loss_func, regnorm=regnorm, tau=tau)
    irgd = erm_irgd(alg, agg_func, n_iter=n_iter2, tol=tol2, verbose=verbose)

    return irgd

def r_regression_irls(Xs, Y, mod, rho_func=None, regnorm=None,
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.0, tol2=1.0e-5, n_iter2=22, 
                      verbose=0, init_param=1):
    """\
    Поиск параметров модели `mod` при помощи классического методо R-регрессии. 
    Параметр `rhofunc` задает функцию от ошибки.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    if rho_func is None:
        _rho_func = funcs.Square()
    else:
        _rho_func = rho_func

    if mod.param is None or init_param:
        mod.init_param()
    if regnorm is not None:
        mod.use_regularizer(regnorm, tau)

    er = erisk(Xs, Y, mod, SquareErrorLoss())
    alg = fg(er, h=h, n_iter=n_iter, tol=tol)
    alg.use_gradient_averager("AdaM2")

    # er2 = erisk(Xs, Y, mod, ErrorLoss(_rho_func), regnorm=regnorm, tau=tau)

    alg.fit()

    E = mod.evaluate(Xs) - Y
    lval = lval_min = _rho_func.evaluate_sum(E)
    lvals = [lval]

    param_min = mod.param.copy()
    weights_min = er.weights.copy()

    for k in range(n_iter2):
        weights = _rho_func.derivative_div_array(E)
        er.use_weights(weights)
        alg.fit()

        E = mod.evaluate(Xs) - Y
        lval = _rho_func.evaluate_sum(E)
        lvals.append(lval)

        if lval < lval_min:
            lval_min = lval
            param_min = mod.param.copy()
            weights_min = er.weights.copy()

    mod.param[:] = param_min
    er.use_weights(weights_min)
    return alg


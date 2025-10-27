#

import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.avragg as avragg
import mlgrad.gd as gd
import mlgrad.weights as weights

from mlgrad.utils import exclude_outliers

from mlgrad.funcs2 import SquareNorm
from mlgrad.loss import SquareErrorLoss, ErrorLoss

from mlgrad import fg, erm_fg, erm_irgd, erisk #mrisk, erisk22

from mlgrad.af import averaging_function

import mlgrad.regr as regr

from .margin_max import MarginMaximization, MarginMaximization2

def classification(X, Y, mod,
               loss_func=loss.MarginLoss(funcs.Hinge(1.0)),
               regnorm=None,
               *,
               weights=None,
               normalizer=None,
               h=0.001, tol=1.0e-6, n_iter=1000, tau=0.001, verbose=0,
               n_restart=1, init_param=1):
    """Поиск оптимальных параметров модели `mod` путем минимизации эмпирического на основе заданной функции
    потерь `lossfunc` для решения задачи классификации.
    Параметры:
        X: 2-мерный массив входных обучающих данных
        Y: 1-мерный массив ожидаемых значений (меток) на выходе
        mod: Модель зависимости для построения разделяющей поверхности между классами
        lossfunc: Функция потерь от ошибок классов (как правило, от величины отступа)
        regnorm: Регуляризатор
        weights: Веса примеров
    """
    return regr.regression(X, Y, mod, loss_func, regnorm, weights=weights, normalizer=normalizer,
               h=h, tol=tol, n_iter=n_iter, tau=tau, verbose=verbose,
               n_restart=n_restart, init_param=init_param)

# classification_as_regr = classification

def classification22(X, Y, mod,
                    lossfunc=loss.MarginMultLoss2(funcs.Square()),
                    regnorm=None,
                    *,
                    weights=None,
                    normalizer=None,
                    h=0.001, tol=1.0e-6, n_iter=1000, tau=0.001, verbose=0, n_restart=1):

    return regr.regression22(X, Y, mod, loss_func, regnorm, weights=weights, normalizer=normalizer,
               h=h, tol=tol, n_iter=n_iter, tau=tau, verbose=verbose,
               n_restart=n_restart, init_param=init_param)
    # if mod.param is None:
    #     mod.init_param()
    # # print(mod.param.base)
    # if regnorm is not None:
    #     mod.use_regularizer(regnorm, tau)

    # er = erisk(Xs, Ys, mod, loss_func)
    # alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter,
    #              normalizer=normalizer,
    #              verbose=verbose, n_restart=n_restart)
    # return alg

# classification_as_regr22 = classification_erm22

def classification_merm_ir(Xs, Y, mod,
                      lossfunc=loss.MarginLoss(funcs.Hinge(0)),
                      agg_func=averaging_function('WM'), regnorm=None, normalizer=None,
                      h=0.001, tol=1.0e-6, n_iter=1000, tau=0.001, tol2=1.0e-6, n_iter2=22, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от отступов.
    
    * `avrfunc` -- усредняющая агрегирующая функция.
    * `lossfunc` -- функция потерь.
    * `Xs` -- входной двумерный массив.
    * `Y` -- массив ожидаемых значений на выходе.
    """
    if mod.param is None:
        mod.init_param()
    if regnorm is not None:
        mod.use_regularizer(regnorm, tau)

    er = erisk(Xs, Y, mod, lossfunc)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)

    if normalizer is not None:
        alg.use_normalizer(normalizer)

    irgd = erm_irgd(alg, agg_func, n_iter=n_iter2, tol=tol2, verbose=verbose)

    return irgd

# classification_m_regr_ir = classification_merm_ir

def m_classification_merm(Xs, Y, mod,
                      lossfunc=loss.NegMargin(),
                      avrfunc=averaging_function('SoftMin', args=(4,)), regnorm=None, normalizer=None,
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от отступов. 

    * `avrfunc` -- усредняющая агрегирующая функция.
    * `lossfunc` -- функция потерь.
    * `Xs` -- входной двумерный массив.
    * `Y` -- массив ожидаемых значений на выходе.
    """
    if mod.param is None:
        mod.init_param()
    if regnorm is not None:
        mod.use_regularizer(regnorm, tau)

    er = erisk(Xs, Y, mod, lossfunc, avg=avrfunc)
    alg = erm_fg(er, n_iter=n_iter, tol=tol, verbose=verbose)

    if normalizer is not None:
        alg.use_normalizer(normalizer)

    return alg

# classification_as_m_regr = m_classification_merm
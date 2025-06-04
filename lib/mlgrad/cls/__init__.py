#

import mlgrad.loss as loss
import mlgrad.funcs as funcs
import mlgrad.avragg as avragg
import mlgrad.gd as gd
import mlgrad.weights as weights
from mlgrad.utils import exclude_outliers

from mlgrad.funcs2 import SquareNorm
from mlgrad.loss import SquareErrorLoss, ErrorLoss

from mlgrad import fg, erm_fg, erm_irgd, erisk, mrisk, erisk22

from mlgrad.af import averaging_function

from .margin_max import MarginMaximization, MarginMaximization2

def classification_erm(Xs, Y, mod, lossfunc=loss.MarginLoss(funcs.Hinge(1.0)), regnorm=None,
               normalizer=None, h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
    if mod.param is None:
        mod.init_param()
    # print(mod.param.base)

    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, 
                 normalizer=normalizer,
                 verbose=verbose, n_restart=n_restart)
    return alg

classification_as_regr = classification_erm

def classification_erm22(Xs, Ys, mod, lossfunc=loss.MarginMultLoss2(funcs.Square()), regnorm=None,
               normalizer=None, h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
    if mod.param is None:
        mod.init_param()
    # print(mod.param.base)

    er = erisk22(Xs, Ys, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, 
                 normalizer=normalizer,
                 verbose=verbose, n_restart=n_restart)
    return alg

classification_as_regr22 = classification_erm22

def classification_merm_ir(Xs, Y, mod, 
                      lossfunc=loss.MarginLoss(funcs.Hinge(0)),
                      avrfunc=averaging_function('WM'), regnorm=None, normalizer=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-6, n_iter2=22, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от отступов. 
    
    * `avrfunc` -- усредняющая агрегирующая функция.
    * `lossfunc` -- функция потерь.
    * `Xs` -- входной двумерный массив.
    * `Y` -- массив ожидаемых значений на выходе.
    """
    if mod.param is None:
        mod.init_param()

    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)

    if normalizer is not None:
        alg.use_normalizer(normalizer)

    wt = weights.MWeights(avrfunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)

    return irgd
    
classification_m_regr_ir = classification_merm_ir

def classification_merm(Xs, Y, mod, 
                      lossfunc=loss.NegMargin(),
                      avrfunc=averaging_function('SoftMax', args=(4,)), regnorm=None, normalizer=None, 
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
    er = mrisk(Xs, Y, mod, lossfunc, avrfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, n_iter=n_iter, tol=tol, verbose=verbose)

    if normalizer is not None:
        alg.use_normalizer(normalizer)
   
    return alg

classification_as_m_regr = classification_merm
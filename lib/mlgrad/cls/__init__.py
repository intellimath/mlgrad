#

import mlgrad.loss as loss
import mlgrad.func as func
import mlgrad.avragg as avragg
import mlgrad.gd as gd
import mlgrad.weights as weights
from mlgrad.utils import exclude_outliers

from mlgrad.func2 import SquareNorm
from mlgrad.loss import SquareErrorLoss, ErrorLoss

from mlgrad import fg, erm_fg, erm_irgd, erisk, mrisk

from mlgrad.af import averaging_function

def classification_as_regr(Xs, Y, mod, lossfunc=loss.MarginLoss(func.Hinge(0)), regnorm=None, 
               h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg


def classification_as_mregr(Xs, Y, mod, 
                      lossfunc=loss.MarginLoss(func.Hinge(0)),
                      avrfunc=averaging_function('WM'), regnorm=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-6, n_iter2=22, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от отступов. 
    
    * `avrfunc` -- усредняющая агрегирующая функция.
    * `lossfunc` -- функция потерь.
    * `Xs` -- входной двумерный массив.
    * `Y` -- массив ожидаемых значений на выходе.
    """
    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)

    wt = weights.MWeights(avrfunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
        
    return irgd

    
def plot_losses_and_errors(alg, Xs, Y, fname=None, logscale=True):
    import numpy as np
    import matplotlib.pyplot as plt

    err = alg.risk.evaluate_losses(Xs)
    err.sort()
    err = err[::-1]
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('Fit curve')
    plt.plot(alg.lvals)
    if logscale:
        plt.gca().set_yscale('log')
    plt.xlabel('step')
    plt.ylabel('mean of losses')
    plt.minorticks_on()
    plt.subplot(1,2,2)
    # err_rat = 1 - sum((1 if y<=1 else 0) for y in err)/len(err)
    # plt.title('Errors %.2f' % err_rat)
    plt.plot(sorted(Y*alg.risk.model.evaluate_all(Xs)), marker='o', markersize='5')
    plt.hlines(0, 0, len(err), color='LightGray')
    plt.minorticks_on()
    plt.grid(1)
    plt.xlabel('error rank')
    plt.ylabel('error value')
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    plt.show()
    
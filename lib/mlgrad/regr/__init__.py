import mlgrad.loss as loss
import mlgrad.func as func
import mlgrad.gd as gd
import mlgrad.weights as weights
from mlgrad.utils import exclude_outliers

from mlgrad.regnorm import SquareNorm
from mlgrad.loss import SquareErrorLoss, ErrorLoss

from mlgrad import fg, erm_fg, erm_irgd, erisk, mrisk

from mlgrad.af import averaging_function

__all__ = 'regression', 'm_regression', 'm_regression_irls', 'r_regression_irls'

def regression(Xs, Y, mod, loss_func=None, regnorm=None, 
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
    er = erisk(Xs, Y, mod, _loss_func, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression(Xs, Y, mod, 
                 loss_func=None, 
                 agg_func=None, regnorm=None, 
                 h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
        
    if loss_func is None:
        _loss_func = SquareErrorLoss()
    else:
        _loss_func = loss_func
    if agg_func is None:
        _agg_func = averaging_function('WM')
    else:
        _agg_func = agg_func
    er = mrisk(Xs, Y, mod, loss_func, _agg_func, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression_irls(Xs, Y, mod, 
                      loss_func=None,
                      agg_func=None, regnorm=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=22, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от ошибок. 
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
    er = erisk(Xs, Y, mod, _loss_func, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)

    wt = weights.MWeights(_agg_func, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
        
    return irgd

def r_regression_irls(Xs, Y, mod, rho_func=None, regnorm=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=22, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи классического методо R-регрессии. 
    Параметр `rhofunc` задает функцию от ошибки.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    loss_func = SquareErrorLoss()
    if rho_func is None:
        _rho_func = func.Sqrt(1.0)
    else:
        _rho_func = rho_func
    er = erisk(Xs, Y, mod, loss_func, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, n_iter=n_iter, tol=tol)
    wt = weights.RWeights(_rho_func, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
    return irgd

def plot_losses_and_errors(alg, Xs, Y, fname=None, logscale=True):
    import numpy as np
    import matplotlib.pyplot as plt

    err = np.abs(Y - alg.risk.model.evaluate_all(Xs))
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
    plt.title('Errors')
    plt.plot(sorted(err), marker='o', markersize='4')
    plt.minorticks_on()
    plt.xlabel('error rank')
    plt.ylabel('error value')
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    plt.show()

def plot_yy(Xs, Y, mod, label, b=0.1):
    import numpy as np 
    import matplotlib.pyplot as plt

    Yp = mod.evaluate_all(Xs)
    E = np.abs(Y - Yp) / np.abs(Y)
    c = sum(E < b) / len(E) * 100
    ymax, ymin = np.max(Y), np.min(Y)
    ypmax, ypmin = np.max(Yp), np.min(Yp)
    ymax = max(ymax, ypmax)
    ymin = min(ymin, ypmin)
    plt.plot([ymin, ymax], [ymin, ymax], color='k', linewidth=0.66)
    plt.fill_between([ymin, ymax], [ymin-b, ymax-b], [ymin+b, ymax+b], color='LightGray')
    plt.scatter(Y, Yp, c='k', s=12, label=r'$\{|err|<%.2f\}\ \to\ %s$ %%' % (b, int(c)))
    plt.title(label)
    plt.ylim(ymin, ymax)
    plt.xlim(ymin, ymax)
    plt.xlabel("original")
    plt.ylabel("predicted")
    plt.legend()

import mlgrad.loss as loss
import mlgrad.func as func
import mlgrad.avragg as avragg
import mlgrad.gd as gd
import mlgrad.weights as weights
from mlgrad.utils import array_exclude_outliers

from mlgrad.regnorm import SquareNorm
from mlgrad.loss import SquareErrorLoss, ErrorLoss

from mlgrad import averager_it, averager_fg, fg, erm_fg, sg, erm_sg, irgd, erm_irgd, erisk, mrisk

__all__ = 'averaging_function', 'regression', 'm_regression', 'm_regression_irls', 'r_regression_irls'

def averaging_function(kind='M', *, rhofunc=func.Sqrt(0.001),
                                 alpha=0.5,
                                 tol=1.0e-8, n_iter=1000):
    """
    Создание экземпляра класса, реализуюего агрегирующую функцию.
    
    Примеры:
    
    # создание сглаженного варианта квантиля с alpha=0.8
    avgfunc = averaging_function('M', func.Sqrt(0.001), alpha=0.8)
    

    # создание варианта цензурированного среднего  alpha=0.8
    avgfunc = averaging_function('WM', func.Sqrt(0.001), alpha=0.8)
    
    # создание сглаженного варианта цензурированного среднего  alpha=0.8
    avgfunc = averaging_function('SWM', func.Sqrt(0.001), alpha=0.8)
    """
    if alpha == 0.5:
        avgfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
    else:
        avgfunc = averager_it(func.QuantileFunc(alpha, rhofunc), tol=tol, n_iter=n_iter)
    
    if kind == 'M':
        avg = avgfunc
    elif kind == 'WM':
        avg = avragg.WMAverage(avgfunc)
    elif kind == 'SWM':
        avg = avragg.ParameterizedAverage(func.WinsorizedSmoothFunc(rhofunc), avgfunc)
    elif kind == 'HM':
        avg = avragg.HMAverage(avgfunc)
    else:
        raise ValueError('Invalid argument value: %s' % kind)
    return avg

def regression(Xs, Y, mod, lossfunc=SquareErrorLoss(), regnorm=None, 
               h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации эмпирического риска. 
    Параметр `lossfunc` задает функцию потерь.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression(Xs, Y, mod, 
                 lossfunc=SquareErrorLoss(), 
                 avrfunc=averaging_function('WM'), regnorm=None, 
                 h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
        
    er = mrisk(Xs, Y, mod, lossfunc, avrfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression_irls(Xs, Y, mod, 
                      lossfunc=SquareErrorLoss(),
                      avrfunc=averaging_function('WM'), regnorm=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=44, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи принципа минимизации агрегирующей функции потерь от ошибок. 
    Параметр `avrfunc` задает усредняющую агрегирующую функцию.
    Параметр `lossfunc` задает функцию потерь.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)

    wt = weights.MWeights(avrfunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
        
    return irgd

def r_regression_irls(Xs, Y, mod, rhofunc=func.Sqrt(1.0), regnorm=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=44, verbose=0):
    """\
    Поиск параметров модели `mod` при помощи классического методо R-регрессии. 
    Параметр `rhofunc` задает функцию от ошибки.
    `Xs` и `Y` -- входной двумерный массив и массив ожидаемых значений на выходе.
    """
    lossfunc = SquareErrorLoss()
    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, n_iter=n_iter, tol=tol)
    wt = weights.RWeights(rhofunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
    return irgd

# def robust_learninig_process(func, Xs, Y, mod, n_last):
#     N = len(Y)
#     Is = tuple(i for i in range(N) if i not in I_exc)
#     Xs1 = Xs[Is]
#     Y1 = Y[Is]
#     alg = func(Xs1, Y1, mod)
    
    
    

def plot_losses_and_errors(alg, Xs, Y, fname=None):
    import numpy as np
    import matplotlib.pyplot as plt

    err = np.abs(Y - alg.risk.model.evaluate_all(Xs))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('Fit curve')
    plt.plot(alg.lvals)
    plt.xlabel('step')
    plt.ylabel('mean of losses')
    plt.minorticks_on()
    plt.subplot(1,2,2)
    plt.title('Errors')
    plt.plot(sorted(err), marker='o', markersize='6')
    plt.minorticks_on()
    plt.xlabel('error rank')
    plt.ylabel('error value')
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    plt.show()

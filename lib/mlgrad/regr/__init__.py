import mlgrad.loss as loss
import mlgrad.func as func
import mlgrad.avragg as avragg
import mlgrad.gd as gd
import mlgrad.weights as weights
from mlgrad.utils import array_exclude_outliers

from mlgrad.regnorm import SquareNorm
from mlgrad.loss import SquareErrorLoss

from mlgrad import averager_it, averager_fg, fg, erm_fg, sg, erm_sg, irgd, erm_irgd, erisk, mrisk

__all__ = 'averaging_function', 'regression', 'm_regression', 'm_regression_irls', 'r_regression_irls'

def averaging_function(kind, **kw):
    if kind == 'M':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        tol = kw.get('tol', 1.0e-8)
        n_iter = kw.get('n_iter', 1000)
        avg = averager_it(rhofunc, tol=tol, n_iter=n_iter)
    elif kind == 'WM':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        tol = kw.get('tol', 1.0e-8)
        n_iter = kw.get('n_iter', 1000)
        avrfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
        avg = avragg.WMAverage(avrfunc)
    elif kind == 'SWM':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        absfunc = kw.get('absfunc', func.Sqrt(0.001))
        tol = kw.get('tol', 1.0e-8)
        n_iter = kw.get('n_iter', 1000)
        avrfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
        avg = avragg.ParameterizedAverage(func.WinsorizedSmoothFunc(absfunc), avrfunc)
    elif kind == 'HM':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        tol = kw.get('tol', 1.0e-8)
        n_iter = kw.get('n_iter', 1000)
        avrfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
        avg = avragg.HMAverage(avrfunc)
    else:
        raise ValueError('Invalid argument value: %s' % kind)
    return avg

def regression(Xs, Y, mod, lossfunc=SquareErrorLoss(), regnorm=SquareNorm(), 
               h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):

    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression(Xs, Y, mod, lossfunc=SquareErrorLoss(), avrfunc=averaging_function('WM'), regnorm=SquareNorm(), 
                 h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, verbose=0, n_restart=1):
        
    er = mrisk(Xs, Y, mod, lossfunc, avrfunc, regnorm=regnorm, tau=tau)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression_irls(Xs, Y, mod, 
                      lossfunc=SquareErrorLoss(), avrfunc=averaging_function('WM'), regnorm=SquareNorm(), 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=21, verbose=0):

    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, tol=tol, n_iter=n_iter)

    wt = weights.MWeights(avrfunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
        
    return irgd

def r_regression_irls(Xs, Y, mod, lossfunc=SquareErrorLoss(), rhofunc=func.Sqrt(1.0), regnorm=SquareNorm(), 
                      h=0.001, tol=1.0e-9, n_iter=1000, tau=0.001, tol2=1.0e-5, n_iter2=21, verbose=0):

    er = erisk(Xs, Y, mod, lossfunc, regnorm=regnorm, tau=tau)
    alg = fg(er, h=h, n_iter=n_iter, tol=tol)
    wt = weights.RWeights(rhofunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2, verbose=verbose)
    return irgd

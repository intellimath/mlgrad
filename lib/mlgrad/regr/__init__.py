#
import mlgrad.model as model
import mlgrad.loss as loss
import mlgrad.func as func
import mlgrad.avragg as avragg
import mlgrad.gd as gd
import mlgrad.regular as regular
import mlgrad.weights as weights
from mlgrad.utils import array_exclude_outliers

from mlgrad import averager_it, averager_fg, fg, erm_fg, sg, erm_sg, irgd, erm_irgd, erisk, mrisk

def averaging_function(kind, **kw):
    if kind == 'M':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        tol = kw.get('tol', 1.0e-6)
        n_iter = kw.get('n_iter', 1000)
        avg = averager_it(rhofunc, tol=tol, n_iter=n_iter)
    elif kind == 'WM':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        tol = kw.get('tol', 1.0e-6)
        n_iter = kw.get('n_iter', 1000)
        avrfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
        avg = avragg.WMAverage(avrfunc)
    elif kind == 'SWM':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        smoothfunc = kw.get('smoothfunc', func.Sqrt(0.001))
        tol = kw.get('tol', 1.0e-6)
        n_iter = kw.get('n_iter', 1000)
        avrfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
        avg = avragg.ParameterizedAverage(func.WinsorizedSmoothFunc(smoothfunc), avrfunc)
    elif kind == 'HM':
        rhofunc = kw.get('rhofunc', func.Quantile_Sqrt(0.5, 0.001))
        tol = kw.get('tol', 1.0e-6)
        n_iter = kw.get('n_iter', 1000)
        avrfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
        avg = avragg.HMAverage(avrfunc)
    else:
        raise ValueError('Invalid argument value: %s' % kind)
    return avg

def regression(Xs, Y, mod, lossfunc=None, h=0.001, tol=1.0e-9, n_iter=1000, verbose=1, n_restart=1):
    _lossfunc = lossfunc
    if lossfunc is None:
        _lossfunc = loss.SquareErrorLoss()
    er = erisk(Xs, Y, mod, _lossfunc)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def m_regression(Xs, Y, mod, lossfunc=None, avrfunc=None, h=0.001, tol=1.0e-9, n_iter=1000, verbose=1, n_restart=1):
    _lossfunc = lossfunc
    if lossfunc is None:
        _lossfunc = loss.SquareErrorLoss()
        
    _avrfunc = avrfunc
    if avrfunc is None:
        _avrfunc = averager_it(func.Quantile_Sqrt(0.5, 0.001))
        _avrfunc = avragg.WMAverage(_avrfunc)

    er = mrisk(Xs, Y, mod, _lossfunc, _avrfunc)
    alg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=n_restart)
    return alg

def r_regression_irls(Xs, Y, mod, lossfunc=None, rhofunc=None, 
                      h=0.001, tol=1.0e-9, n_iter=1000, tol2=1.0e-5, n_iter2=21):
    _lossfunc = lossfunc
    if lossfunc is None:
        _lossfunc = loss.SquareErrorLoss()

    _rhofunc = rhofunc
    if rhofunc is None:
        _rhofunc = func.Sqrt(1.0)
    er = erisk(Xs, Y, mod, _lossfunc)
    alg = fg(er, h=h, n_iter=n_iter, tol=tol)
    wt = weights.RWeights(_rhofunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2)
    return irgd

def m_regression_irls(Xs, Y, mod, 
                      lossfunc=None, avrfunc=None, h=0.001, 
                      tol=1.0e-9, n_iter=1000, tol2=1.0e-5, n_iter2=21):
    _lossfunc = lossfunc
    if lossfunc is None:
        _lossfunc = loss.SquareErrorLoss()

    er = erisk(Xs, Y, mod, _lossfunc)
    alg = fg(er, h=h, tol=tol)

    _avrfunc = avrfunc
    if avrfunc is None:
        _avrfunc = averager_it(func.Quantile_Sqrt(0.5, 0.001))
        _avrfunc = avragg.WMAverage(_avrfunc)

    wt = weights.MWeights(_avrfunc, er)
    irgd = erm_irgd(alg, wt, n_iter=n_iter2, tol=tol2)
        
    return irgd

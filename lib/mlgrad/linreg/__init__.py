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

def regression(Xs, Y, mod, lossfunc=None, h=0.001, tol=1.0e-8, n_iter=1000, verbose=1, n_restart=5):
    _lossfunc = lossfunc
    if lossfunc is None:
        _lossfunc = loss.SquareErrorLoss()
    er = erisk(Xs, Y, mod, _lossfunc)
    alg_fg = erm_fg(er, h=h, tol=tol, n_iter=n_iter, verbose=verbose, n_restart=5)
    return alg_fg


def m_regression(Xs, Y, mod, 
                 lossfunc=None, avrfunc=None, h=0.001, 
                 tol=1.0e-8, n_iter=1000, tol2=1.0e-5, n_iter2=21):
    _lossfunc = lossfunc
    if lossfunc is None:
        _lossfunc = loss.SquareErrorLoss()

    er = erisk(Xs, Y, mod, _lossfunc)
    alg_fg = fg(er, h=h, tol=tol)

    _avrfunc = avrfunc
    if avrfunc is None:
        _avrfunc = averager_it(func.Quantile_Sqrt(0.5, 0.001))
        _avrfunc = avragg.WMAverage(_avrfunc)

    wg = weights.MWeights(_avrfunc, er)
    irgd = erm_irgd(alg_fg, wg, n_iter=n_iter2, tol=tol2)
        
    return irgd

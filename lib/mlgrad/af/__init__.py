import mlgrad.func as func
import mlgrad.avragg as avragg

__all__ = ['averaging_function']

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
        avgfunc = avragg.MAverage(rhofunc, tol=tol, n_iter=n_iter)
        # avgfunc = averager_it(rhofunc, tol=tol, n_iter=n_iter)
    else:
        avgfunc = avragg.MAverage(func.quantile_func(alpha, rhofunc), tol=tol, n_iter=n_iter)
        # avgfunc = averager_it(func.QuantileFunc(alpha, rhofunc), tol=tol, n_iter=n_iter)
    
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
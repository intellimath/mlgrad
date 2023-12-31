import mlgrad.func as func
import mlgrad.avragg as avragg

__all__ = ['averaging_function']

def averaging_function(
             kind='M',
             rhofunc=func.Sqrt(0.001),
             *,
             tol=1.0e-8, n_iter=1000):
    """
    Создание экземпляра класса, реализуюего агрегирующую функцию.
    
    Примеры:
    
    # создание сглаженного варианта квантиля с alpha=0.8
    avgfunc = averaging_function('M', func.quantile_func(0.8, func.Sqrt(0.001)))    

    # создание варианта цензурированного среднего  alpha=0.8
    avgfunc = averaging_function('WM', func.quantile_func(0.8, func.Sqrt(0.001)))
    
    # создание сглаженного варианта цензурированного среднего  alpha=0.8
    avgfunc = averaging_function('SWM', func.quantile_func(0.8, func.Sqrt(0.001)))
    """
    avgfunc = avragg.MAverage(rhofunc, tol=tol, n_iter=n_iter)

    if kind == 'M':
        avg = avgfunc
    elif kind == 'WM':
        avg = avragg.WMAverage(avgfunc)
    elif kind == 'SWM':
        avg = avragg.ParameterizedAverage(func.WinsorizedSmoothFunc(rhofunc), avgfunc)
    elif kind == 'HM':
        avg = avragg.HMAverage(avgfunc)
    elif kind == 'AM':
        avg = avragg.ArithMean()
    else:
        raise ValueError('Invalid argument value: %s' % kind)
    return avg


def scaling_function(
             kind='S',
             rhofunc=func.Sqrt(0.001),
             *,
             tol=1.0e-8, n_iter=200):

    avgfunc = avragg.SAverage(rhofunc, tol=tol, n_iter=n_iter)

    if kind == 'S':
        avg = avgfunc
    else:
        raise ValueError('Invalid argument value: %s' % kind)
    return avg
    
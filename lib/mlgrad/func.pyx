# coding: utf-8
 
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False

# The MIT License (MIT)
#
# Copyright © «2015–2021» <Shibzukhov Zaur, szport at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from libc.math cimport fabs, pow, sqrt, fmax, exp, log, atan
from libc.math cimport isnan, isinf
from libc.stdlib cimport strtod

import numpy as np

cimport cython

cdef double c_nan = strtod("NaN", NULL)
cdef double c_inf = strtod("Inf", NULL)

cdef dict _func_table = {}
def register_func(cls, tag):
    _func_table[tag] = cls
    return cls

def func_from_dict(ob):
    f = _func_table[ob['name']]
    return f(*ob['args'])

cdef class Func(object):
    #
    cdef double evaluate(self, const double x) nogil:
        return 0.
    #
    def __call__(self, x):
        cdef double v = x
        return self.evaluate(v)
    #
    def __getitem__(self, x):
        cdef double v = x
        return self.derivative(v)
    #
    cdef double derivative(self, const double x) nogil:
        return 0.
    #
    cdef double derivative2(self, const double x) nogil:
        return 0.
    #
    cdef double derivative_div_x(self, const double x) nogil:
        return self.derivative(x) / x

cdef class Comp(Func):
    #
    def __init__(self, Func f, Func g):
        self.f = f
        self.g = g
    #
    cdef double evaluate(self, const double x) nogil:
        return self.f.evaluate(self.g.evaluate(x))
    #
    cdef double derivative(self, const double x) nogil:
        return self.f.derivative(self.g.evaluate(x)) * self.g.derivative(x)
    #
    cdef double derivative2(self, const double x) nogil:
        cdef double dg = self.g.derivative(x)
        cdef double y = self.g.evaluate(x)
        
        return self.f.derivative2(y) * dg * dg + \
               self.f.derivative(y) * self.g.derivative2(x)
    #
    cdef double derivative_div_x(self, const double x) nogil:
        return self.f.derivative(self.g.evaluate(x)) * self.g.derivative_div_x(x)
    
    def to_dict(self):
        return { 'name':'comp',
                'args': (self.f.to_dict(), self.g.to_dict() ) 
               }

cdef class CompSqrt(Func):
    #
    def __init__(self, Func f):
        self.f = f
    #
    cdef double evaluate(self, const double x) nogil:
        return self.f.evaluate(sqrt(x))
    #
    cdef double derivative(self, const double x) nogil:
        return 0.5 * self.f.derivative(sqrt(x)) / sqrt(x)
    #
    cdef double derivative2(self, const double x) nogil:
        cdef double dg = 0.5 / sqrt(x)
        cdef double y = sqrt(x)
        
        return self.f.derivative2(y) * dg * dg + \
               -0.25 * self.f.derivative(y) / (x*y)
    #
    cdef double derivative_div_x(self, const double x) nogil:
        return 0.5 * self.f.derivative(sqrt(x)) / (x*sqrt(x))
    
    def to_dict(self):
        return { 'name':'compsqrt',
                'args': (self.f.to_dict(), self.g.to_dict() ) 
               }

cdef class ZeroOnPositive(Func):
    #
    def __init__(self, Func f):
        self.f = f
    #
    cdef double evaluate(self, const double x) nogil:
        if x > 0:
            return 0
        else:
            return self.f.evaluate(x)
    #
    cdef double derivative(self, const double x) nogil:
        if x > 0:
            return 0
        else:
            return self.f.derivative(x)
    #
    cdef double derivative2(self, const double x) nogil:
        if x > 0:
            return 0
        else:
            return self.f.derivative2(x)
    #
    cdef double derivative_div_x(self, const double x) nogil:
        if x > 0:
            return 0
        else:
            return self.f.derivative_div_x(x)

cdef class FuncExp(Func):
    #
    def __init__ (self, Func f):
        self.f = f
    #
    cdef double evaluate(self, const double x) nogil:
        return self.f.evaluate(exp(x))
    #
    cdef double derivative(self, const double x) nogil:
        cdef double y = exp(x)
        return self.f.derivative(y) * y
    #
    cdef double derivative2(self, const double x) nogil:
        cdef double y = exp(x)
        return (self.f.derivative(y) + self.f.derivative2(y) * y) * y
        
cdef class Id(Func):
    #
    cdef double evaluate(self, const double x) nogil:
        return x
    #
    cdef double derivative(self, const double x) nogil:
        return 1
    #
    cdef double derivative2(self, const double x) nogil:
        return 0
    #
    def _repr_latex_(self):
        return '$\mathrm{id}(x)=x$'

cdef class QuantileFunc(Func):
    #
    def __init__(self, alpha, Func func):
        self.alpha = alpha
        self.f = func
    #
    cdef double evaluate(self, const double x) nogil:
        if x > 0:
            return self.alpha * self.f.evaluate(x)
        elif x < 0:
            return (1-self.alpha) * self.f.evaluate(x)
        else:
            return 0.5 * self.f.evaluate(0)
    #
    cdef double derivative(self, const double x) nogil:
        if x > 0:
            return self.alpha * self.f.derivative(x)
        elif x < 0:
            return (1-self.alpha) * self.f.derivative(x)
        else:
            return 0.5 * self.f.derivative(0)
    #
    cdef double derivative2(self, const double x) nogil:
        if x > 0:
            return self.alpha * self.f.derivative2(x)
        elif x < 0:
            return (1-self.alpha) * self.f.derivative2(x)
        else:
            return 0.5 * self.f.derivative2(0)
    #
    cdef double derivative_div_x(self, const double x) nogil:
        if x > 0:
            return self.alpha * self.f.derivative_div_x(x)
        elif x < 0:
            return (1-self.alpha) * self.f.derivative_div_x(x)
        else:
            return 0.5 * self.f.derivative_div_x(0)
    #
    def _repr_latex_(self):
        return '$\mathrm{id}(x)=x$'
    
    def to_dict(self):
        return { 'name':'quantile_func',
                'args': (self.alpha, self.f.to_dict() ) 
               }
    

cdef class Neg(Func):
    #
    cdef double evaluate(self, const double x) nogil:
        return -x
    #
    cdef double derivative(self, const double x) nogil:
        return -1
    #
    cdef double derivative2(self, const double x) nogil:
        return 0
    #
    def _repr_latex_(self):
        return '$\mathrm{id}(x)=-x$'

cdef class ModSigmoidal(Func):
    #
    def __init__(self, a=1):
        self.label = u'σ'
        self.a = a
    #
    cdef double evaluate(self, const double x) nogil:
        return x / (self.a + fabs(x))
    #
    cdef double derivative(self, const double x) nogil:
        cdef double v = (self.a + fabs(x))
        return self.a / (v*v)
    #
    cdef double derivative2(self, const double x) nogil:
        cdef double v = (self.a + fabs(x))
        if x > 0:
            return -2.0 * self.a / v*v*v
        elif x < 0: 
            return 2.0 * self.a / v*v*v
        else:
            return 0
    #
    def _repr_latex_(self):
        return '$%s(x, a)=\dfrac{x}{a+|x|}$' % self.label
        
cdef class Sigmoidal(Func):
    #
    def __init__(self, p=1):
        self.label = u'σ'
        self.p = p
    #
    @cython.cdivision(True)
    cdef double evaluate(self, const double x) nogil:
        return 1.0/(1.0 + exp(-self.p * x))
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double p = self.p
        cdef double v = 1.0/(1.0 + exp(-p * x))
        return p * v * (1.0 - v)
    #
    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double p = self.p
        cdef double v = 1.0/(1.0 + exp(-p * x))
        return v*(1-v)*(1-2*v)*p*p
    #
    def _repr_latex_(self):
        return '$%s(x, p)=\dfrac{1}{1+e^{-px}}$' % self.label

    def to_dict(self):
        return { 'name':'sigmoidal', 
                 'args': (self.p,) }
    
cdef class Arctang(Func):
    #
    def __init__(self, a=1):
        self.label = u'σ'
        self.a = a
    #
    @cython.cdivision(True)
    cdef double evaluate(self, const double x) nogil:
        return atan(x/self.a)
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double v = x/self.a
        return 1 / (self.a * (1 + v*v))
    #
    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double v = x /self.a
        cdef double a2 = self.a * self.a
        cdef double u = 1 + v*v
        return -2*v / (a2 * u*u)
    #
    def _repr_latex_(self):
        return '$%s(x, p)=\dfrac{1}{1+e^{-px}}$' % self.label

    def to_dict(self):
        return { 'name':'arctg', 
                 'args': (self.a,) }
    
cdef class Softplus(Func):
    #
    def __init__(self, a=1):
        self.label = u'softplus'
        self.a = a
        if a == 1:
            self.log_a = 0
        else:
            self.log_a = log(a)
    #
    cdef double evaluate(self, const double x) nogil:
        return log(self.a + exp(x)) - self.log_a
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double v = exp(x)
        return v / (self.a + v)
    #
    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double v1 = exp(x)
        cdef double v2 = self.a + v1
        return self.a * v1 / v2*v2
    #
    def _repr_latex_(self):
        return '$%s(x, a)=\ln(a+e^x)$' % self.label
        
    def to_dict(self):
        return { 'name':'softplus', 
                 'args': (self.a,) }

cdef class Threshold(Func):
    #
    def __init__(self, theta=0):
        self.label = u'H'
        self.theta = theta
    #
    cdef double evaluate(self, const double x) nogil:
        if x >= self.theta:
            return 1
        else:
            return 0
    #
    cdef double derivative(self, const double x) nogil:
        if x == self.theta:
            return c_inf
        else:
            return 0
    #
    cdef double derivative2(self, const double x) nogil:
        return c_nan
    #
    def _repr_latex_(self):
        return '$%s(x, \theta)=\cases{1&x\geq\theta\\0&x<0}$' % self.label

    def to_dict(self):
        return { 'name':'threshold', 
                 'args': (self.theta,) }
    
cdef class Sign(Func):
    #
    def __init__(self, theta=0):
        self.label = u'sign'
        self.theta = theta
    #
    cdef double evaluate(self, const double x) nogil:
        if x > self.theta:
            return 1
        elif x < self.theta:
            return -1
        else:
            return 0
    #
    cdef double derivative(self, const double x) nogil:
        if x == self.theta:
            return c_inf
        else:
            return 0
    #
    cdef double derivative2(self, const double x) nogil:
        return c_nan
    #
    def _repr_latex_(self):
        return '$%s(x, \theta)=\cases{1&x\geq\theta\\0&x<0}$' % self.label
        
    def to_dict(self):
        return { 'name':'sign', 
                 'args': (self.theta,) }

cdef class Quantile(Func):
    #
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    #
    cdef double evaluate(self, const double x) nogil:
        if x < 0:
            return (self.alpha - 1) * x
        elif x > 0:
            return self.alpha * x
        else:
            return 0
    #
    cdef double derivative(self, const double x) nogil:
        if x < 0:
            return self.alpha - 1.0
        elif x > 0:
            return self.alpha
        else:
            return 0
    #
    cdef double derivative2(self, const double x) nogil:
        if x == 0:
            return c_inf
        else: 
            return 0
    #
    def _repr_latex_(self):
        return r"$ρ(x)=(\alpha - [x < 0])x$"

    def to_dict(self):
        return { 'name':'quantile', 
                 'args': (self.alpha,) }

cdef class Expectile(Func):
    #
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    #
    cdef double evaluate(self, const double x) nogil:
        if x < 0:
            return 0.5 * (1. - self.alpha) * x * x
        elif x > 0:
            return 0.5 * self.alpha * x * x
        else:
            return 0
    #
    cdef double derivative(self, const double x) nogil:
        if x < 0:
            return (1.0 - self.alpha) * x
        elif x > 0:
            return self.alpha * x
        else:
            return 0
    #
    cdef double derivative2(self, const double x) nogil:
        if x < 0:
            return (1.0 - self.alpha)
        elif x > 0:
            return self.alpha
        else:
            return 0.5
    #
    cdef double derivative_div_x(self, const double x) nogil:
        if x < 0:
            return (1.0 - self.alpha)
        elif x > 0:
            return self.alpha
        else:
            return 0.5
    #
    def _repr_latex_(self):
        return r"$ρ(x)=(\alpha - [x < 0])x|x|$"

    def to_dict(self):
        return { 'name':'expectile', 
                 'args': (self.alpha,) }

cdef class Power(Func):
    #
    def __init__(self, p=2.0, alpha=0):
        self.p = p
        self.alpha = alpha
        self.alpha_p = pow(self.alpha, self.p)
    #
    cdef double evaluate(self, const double x) nogil:
        return pow(fabs(x) + self.alpha, self.p) / self.p
    #
    cdef double derivative(self, const double x) nogil:
        cdef double val
        val = pow(fabs(x) + self.alpha, self.p-1) 
        if x < 0:
            val = -val
        return val
    #
    cdef double derivative2(self, const double x) nogil:
        return (self.p-1) * pow(fabs(x) + self.alpha, self.p-2)
    #
    cdef double derivative_div_x(self, const double x) nogil:
        return pow(fabs(x) + self.alpha, self.p-2)
    #
    def _repr_latex_(self):
        return r"$ρ(x)=\frac{1}{p}(|x|+\alpha)^p$"

    def to_dict(self):
        return { 'name':'power', 
                 'args': (self.p, self.alpha,) }
    
cdef class Square(Func):
    #
    cdef double evaluate(self, const double x) nogil:
        return 0.5 * x * x
    #
    cdef double derivative(self, const double x) nogil:
        return x
    #
    cdef double derivative_div_x(self, const double x) nogil:
        return 1
    #
    cdef double derivative2(self, const double x) nogil:
        return 1
    #
    def _repr_latex_(self):
        return r"$ρ(x)=0.5x^2$"

    def to_dict(self):
        return { 'name':'square', 
                 'args': () }
    
cdef class SquareSigned(Func):
    #
    cdef double evaluate(self, const double x) nogil:
        cdef double val = 0.5 * x * x
        if x >= 0:
            return val
        else:
            return -val
    #
    cdef double derivative(self, const double x) nogil:
        return fabs(x)
    #
    cdef double derivative2(self, const double x) nogil:
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        else:
            return 0.
    #
    def _repr_latex_(self):
        return r"$ρ(x)=0.5x^2$"

cdef class Absolute(Func):
    #
    cdef double evaluate(self, const double x) nogil:
        return fabs(x)
    #
    cdef double derivative(self, const double x) nogil:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            0
    #
    cdef double derivative2(self, const double x) nogil:
        if x == 0:
            return c_inf
        else:
            return 0
    #
    def _repr_latex_(self):
        return r"$ρ(x)=|x|$"

    def to_dict(self):
        return { 'name':'absolute', 
                 'args': () }
    
cdef class Quantile_AlphaLog(Func):
    #
    def __init__(self, alpha=1.0, q=0.5):
        assert alpha > 0
        self.alpha = alpha
        self.q = q
        if alpha == 0:
            self.alpha2 = 0.
        else:
            self.alpha2 = self.alpha*log(self.alpha)
    #
    cdef double evaluate(self, const double x) nogil:
        cdef double val
        if x < 0:
            val = -x - self.alpha*log(self.alpha - x) + self.alpha2
            return (1.0-self.q) * val
        elif x > 0:
            val = x - self.alpha*log(self.alpha + x) + self.alpha2
            return self.q * val
        else:
            return 0
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double val
        if x < 0:
            val = x / (self.alpha - x)            
            return (1-self.q) * val
        elif x > 0:
            val = x / (self.alpha + x)            
            return self.q * val
        else:
            return self.q - 0.5
    #
    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double v
        if x < 0:
            v = self.alpha - x
            return (1-self.q)*self.alpha / (v*v)
        elif x > 0:
            v = self.alpha + x
            return self.q*self.alpha / (v*v)
        else:
            return 0.5 / self.alpha
    #
    @cython.cdivision(True)
    cdef double derivative_div_x(self, const double x) nogil:
        cdef double val
        if x < 0:
            val = 1 / (self.alpha - x)
            return (1-self.q) * val
        elif x > 0:
            val = 1 / (self.alpha + x)
            return self.q * val
        else:
            return (self.q - 0.5) / self.alpha
    #
    def _repr_latex_(self):
        return r"$ρ_q(x)=\mathrm{sign}_q(x)(|x| - \alpha\ln(\alpha+|x|)+\alpha\ln\alpha)$"

    def to_dict(self):
        return { 'name':'quantile_alpha_log', 
                 'args': (self.alpha, self.q) }
    
cdef class Logistic(Func):

    def __init__(self, p=1.0):
        assert p > 0
        self.p = p

    @cython.cdivision(True)
    cdef double evaluate(self, const double x) nogil:
        return log(1.0 + exp(fabs(x) / self.p)) - log(2)

    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double v = exp(fabs(x) / self.p)
        if x > 0:
            return v / (1.0 + v) / self.p
        elif x < 0:
            return -v / (1.0 + v) / self.p
        else:
            return 0

    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double v = exp(fabs(x) / self.p)
        if x > 0:
            return 1 / ((1 + v) * (1+v)) / self.p
        elif x < 0:
            return -1 / ((1 + v) * (1+v)) / self.p
        else:
            return 0 

    def _repr_latex_(self):
        return r"$\ell(y,\tilde y)=\log(1+e^{|y-\tilde y|/p})$"

    def to_dict(self):
        return { 'name':'logistic', 
                 'args': (self.p,) }
    
cdef class Hinge(Func):
    #
    def __init__(self, C=1.0):
        self.C = C
    #
    cdef double evaluate(self, const double x) nogil:
        if x >= self.C:
            return 0
        else:
            return self.C - x
    #
    cdef double derivative(self, const double x) nogil:
        if x >= self.C:
            return 0
        else:
            return -1
    #
    cdef double derivative2(self, const double x) nogil:
        if x >= self.C:
            return -c_inf
        else:
            return 0
    #
    def _repr_latex_(self):
        return r"$ρ(x)=(c-x)_{+}$"

    def to_dict(self):
        return { 'name':'hinge', 
                 'args': (self.C,) }
    
cdef class HingeSqrt(Func):
    #
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.alpha2 = alpha*alpha
    #
    cdef double evaluate(self, const double x) nogil:
        return -x + sqrt(self.alpha2 + x*x)
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        return -1 + x/sqrt(self.alpha2 + x*x)
    #
    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        return 1./sqrt(self.alpha2 + x*x)
    #
    def _repr_latex_(self):
        return r"$ρ(x)=-x + \sqrt{c^2+x^2}$"

    def to_dict(self):
        return { 'name':'hinge_sqrt', 
                 'args': (self.alpha,) }
    
cdef class Huber(Func):

    def __init__(self, C=1.345):
        self.C = C

    @cython.cdivision(True)
    cdef double evaluate(self, const double x) nogil:
        cdef double x_abs = fabs(x)
        
        if x_abs > self.C:
            return x_abs - 0.5 * self.C
        else:
            return 0.5 * x*x / self.C

    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double x_abs = fabs(x)

        if x > self.C:
            return 1.
        elif x < -self.C:
            return -1.
        else:
            return x / self.C
    #
    @cython.cdivision(True)
    cdef double derivative_div_x(self, const double x) nogil:
        cdef double x_abs = fabs(x)

        if x_abs > self.C:
            return 1. / x_abs
        else:
            return 1. / self.C

    def _repr_latex_(self):
        return r"""$\displaystyle
            \rho(x)=\cases{
                0.5x^2/C, & |x|<C\\
                |x|-0.5C, & |x| \geq C
            }
        $"""

    def to_dict(self):
        return { 'name':'huber', 
                 'args': (self.C,) }
    
cdef class TM(Func):
    #
    def __init__(self, a=1):
        self.a = a
    #
    cdef double evaluate(self, const double x) nogil:
        if x <= 0:
            return x*x/2
        else:
            return self.a * x
    #
    cdef double derivative(self, const double x) nogil:
        if x <= 0:
            return x
        else:
            return self.a
    #
    cdef double derivative2(self, const double x) nogil:
        if x <= 0:
            return 1
        else:
            return 0
    #
    cdef double derivative_div_x(self, const double x) nogil:
        if x <= 0:
            return 1
        else:
            return self.a / x
    #
    def _repr_latex_(self):
        return r"""$\displaystyle
            \rho(x)=\cases{
                frac{1}{2}x^2, & x<0\\
                ax, & x\geq 0
            }
        $"""

cdef class LogSquare(Func):
    
    def __init__(self, a=1.0):
        self.a = a
        self.a2 = a * a
    
    cdef double evaluate(self, const double x) nogil:
        return log(self.a2 + x*x)
    
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        return 2 * x / (self.a2 + x*x)

    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double x2 = x*x
        cdef double xa = self.a + x2 
        return 2 * (self.a2 - x2) / xa * xa
    
    def _repr_latex_(self):
        return r'$\ln(a^2 + x^2)$'
    
    def to_dict(self):
        return { 'name':'log_square', 
                 'args': (self.a,) }
    
cdef class Tukey(Func):

    def __init__(self, C=4.685):
        self.C = C
        self.C2 = C * C / 6.

    cdef double evaluate(self, const double x) nogil:
        cdef double v = x/self.C
        cdef double v2 = v*v
        cdef double v3 = 1 - v2
        
        if v <= self.C:
            return self.C2 * (1 - v3*v3*v3)
        else:
            return self.C2

    cdef double derivative(self, const double x) nogil:
        cdef double v = x/self.C
        cdef double v3 = 1 - v*v
        
        if v <= self.C:
            return x * v3*v3
        else:
            return 0

    cdef double derivative2(self, const double x) nogil:
        cdef double v = x/self.C
        cdef double v3 = 1 - v*v
        
        if v <= self.C:
            return v3*v3 - 4*v3*v*v
        else:
            return 0

    def _repr_latex_(self):
        return r"""$\displaystyle
            \pho(x)=\cases{
                (C^2/6) (1-[1-(x/C)^2]^3), & |x|\leq C\\
                C^2/6, & |x| > C
            }
        $"""

    def to_dict(self):
        return { 'name':'tukey', 
                 'args': (self.C,) }

cdef class SoftAbs(Func):
    #
    def __init__(self, eps=1.0):
        self.eps = eps
    #
    @cython.cdivision(True)
    cdef double evaluate(self, const double x) nogil:
        return x * x / (self.eps + fabs(x))
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double v = self.eps + fabs(x)
        return x * (self.eps + v) / (v * v)
    #
    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double eps = self.eps
        cdef double v = eps + fabs(x)
        return 2 * eps * eps / (v * v * v)
    #
    @cython.cdivision(True)
    cdef double derivative_div_x(self, const double x) nogil:
        cdef double v = self.eps + fabs(x)
        return (self.eps + v) / (v * v)
    #
    def _repr_latex_(self):
        return r"$p(x)=\frac{x^2}{\varepsilon+|x|}$"
    
    def to_dict(self):
        return { 'name':'softabs', 
                 'args': (self.eps,) }

    
cdef class Sqrt(Func):
    #
    def __init__(self, eps=1.0, zero=False):
        self.eps = eps
        self.eps2 = eps*eps
        self.zero = 0. if zero else eps
#         self.alpha = alpha
    #
    cdef double evaluate(self, const double x) nogil:
        return sqrt(self.eps2 + x*x) - self.eps
    #
    @cython.cdivision(True)
    @cython.final
    cdef double derivative(self, const double x) nogil:
        cdef double v = self.eps2 + x*x
        return x / sqrt(v)
    #
    @cython.cdivision(True)
    @cython.final
    cdef double derivative2(self, const double x) nogil:
        cdef double v = self.eps2 + x*x
        return self.eps2 / (v * sqrt(v))
    #
    @cython.cdivision(True)
    @cython.final
    cdef double derivative_div_x(self, const double x) nogil:
        return 1. / sqrt(self.eps2 + x*x)
    #
    def _repr_latex_(self):
        return r"$p(x)=\sqrt{\varepsilon^2+x^2}$"
    
    def to_dict(self):
        return { 'name':'sqrt', 
                 'args': (self.eps) }

cdef class Quantile_Sqrt(Func):
    #
    def __init__(self, alpha=0.5, eps=1.0):
        self.alpha = alpha
        self.eps = eps
        self.eps2 = eps*eps
    #
    cdef double evaluate(self, const double x) nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return (sqrt(v) - self.eps) * self.alpha
        else:
            return (sqrt(v) - self.eps) * (1-self.alpha)
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x) nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return self.alpha * x / sqrt(v)
        else:
            return (1.-self.alpha) * x / sqrt(v)
    #
    @cython.cdivision(True)
    cdef double derivative2(self, const double x) nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return self.alpha * self.eps2 / (v * sqrt(v))
        else:
            return (1.-self.alpha) * self.eps2 / (v * sqrt(v))
    #
    @cython.cdivision(True)
    cdef double derivative_div_x(self, const double x) nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return self.alpha / sqrt(v)
        else:
            return (1.-self.alpha) / sqrt(v)
    #
    def _repr_latex_(self):
        return r"$p(x)=(\sqrt{\varepsilon^2+x^2}-\varepsilon)_\alpha$"
    
    def to_dict(self):
        return { 'name':'quantile_sqrt', 
                 'args': (self.alpha, self.eps) }
    
    
cdef class Exp(Func):
    #
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    #
    cdef double evaluate(self, const double x) nogil:
        return exp(x/self.alpha)
    #
    cdef double derivative(self, const double x) nogil:
        return exp(x/self.alpha)/self.alpha
    #
    cdef double derivative2(self, const double x) nogil:
        return exp(x/self.alpha)/self.alpha/self.alpha
    #
    def _repr_latex_(self):
        return r"$\rho(x)=\exp{x/\alpha}$"

    def to_dict(self):
        return { 'name':'exp', 
                 'args': (self.alpha,) }
    
cdef class Log(Func):
    #
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    #
    cdef double evaluate(self, const double x) nogil:
        return log(self.alpha+x)
    #
    cdef double derivative(self, const double x) nogil:
        return 1 / (self.alpha+x)
    #
    cdef double derivative2(self, const double x) nogil:
        cdef double x2 = self.alpha+x
        return -1 / (x2*x2)
    #
    def _repr_latex_(self):
        return r"$\rho(x)=\ln{\alpha+x}$"

    def to_dict(self):
        return { 'name':'log', 
                 'args': (self.alpha,) }
    
cdef class ParameterizedFunc:
    #
    def __call__(self, x, u):
        return self.evaluate(x, u)
    #
    cdef double evaluate(self, const double x, const double u) nogil:
        return 0
    #
    cdef double derivative(self, const double x, const double u) nogil:
        return 0
    #
    cdef double derivative_u(self, const double x, const double u) nogil:
        return 0

cdef class WinsorizedFunc(ParameterizedFunc):
    # 
    cdef double evaluate(self, const double x, const double u) nogil:
        if x > u:
            return u
        elif x < -u:
            return -u
        else:
            return x
    #
    cdef double derivative(self, const double x, const double u) nogil:
        if x > u or x < -u:
            return 0
        else:
            return 1
    #
    cdef double derivative_u(self, const double x, const double u) nogil:
        if x > u or x < -u:
            return 1
        else:
            return 0
        
    def to_dict(self):
        return { 'name':'winsorized', 
                 'args': () }
        

cdef class SoftMinFunc(ParameterizedFunc):
    #
    def __init__(self, a = 1):
        self.a = a
    #
    @cython.cdivision(True)
    cdef double evaluate(self, const double x, const double u) nogil:
        if u < x:
            return u - log(1. + exp(-self.a*(x-u))) / self.a
        else:
            return x - log(1. + exp(-self.a*(u-x))) / self.a
    #
    @cython.cdivision(True)
    cdef double derivative(self, const double x, const double u) nogil:
        return 1. / (1. + exp(-self.a*(u-x)))
    #
    @cython.cdivision(True)
    cdef double derivative_u(self, const double x, const double u) nogil:
        return 1. / (1. + exp(-self.a*(x-u)))

    def to_dict(self):
        return { 'name':'softmin', 
                 'args': (self.a,) }
    
cdef class  WinsorizedSmoothFunc(ParameterizedFunc):
    # 
    def __init__(self, Func f):
        self.f = f
    #
    cdef double evaluate(self, const double x, const double u) nogil:
        return 0.5 * (x + u - self.f.evaluate(x - u))
    #
    cdef double derivative(self, const double x, const double u) nogil:
        return 0.5 * (1. - self.f.derivative(x - u))
    #
    cdef double derivative_u(self, const double x, const double u) nogil:
        return 0.5 * (1. + self.f.derivative(x - u))

    def to_dict(self):
        return { 'name':'winsorized_soft', 
                 'args': (self.f.to_dict(),) }
    
cdef class KMinSquare(Func):
    #
    def __init__(self, c):
        self.c = np.asarray(c, 'd')
        self.n_dim = c.shape[0]
        self.j_min = 0
    #
    cdef double evaluate(self, const double x) nogil:
        cdef int j, j_min, n_dim = self.n_dim
        cdef double d, d_min
    
        d_min = self.c[0]
        j_min = 0
        j = 1
        while j < n_dim:
            d = self.c[j]
            if fabs(x - d) < d_min:
                j_min = j
                d_min = d
            j += 1
        self.j_min = j_min
        return 0.5 * (x - d_min) * (x - d_min)
    #
    cdef double derivative(self, const double x) nogil:
        return x - self.c[self.j_min]
    #
    cdef double derivative2(self, const double x) nogil:
        return 1
    #
    def _repr_latex_(self):
        return r"$\rho(x)=\min_{j=1,\dots,q} (x-c_j)^2/2$"

    def to_dict(self):
        return { 'name':'kmin_square', 
                 'args': (self.c.tolist(),) }

    
register_func(Comp, 'comp')
register_func(QuantileFunc, 'quantile_func')
register_func(Sigmoidal, 'sigmoidal')
register_func(KMinSquare, 'kmin_square')
register_func(WinsorizedSmoothFunc, 'winsorized_smooth')
register_func(SoftMinFunc, 'softmin')
register_func(WinsorizedFunc, 'winsorized')
register_func(Log, 'log')
register_func(Exp, 'exp')
register_func(Quantile_Sqrt, 'quantile_sqrt')
register_func(Sqrt, 'sqrt')
register_func(SoftAbs, 'softabs')
register_func(Tukey, 'tukey')
register_func(LogSquare, 'log_square')
register_func(Huber, 'huber')
register_func(HingeSqrt, 'hinge_sqrt')
register_func(Hinge, 'hinge')
register_func(Logistic, 'logistic')
register_func(Quantile_AlphaLog, 'quantile_alpha_log')
register_func(Absolute, 'absolute')
register_func(Square, 'square')
register_func(Power, 'power')
register_func(Expectile, 'expectile')
register_func(Quantile, 'quantile')
register_func(Sign, 'sign')
register_func(Threshold, 'threshold')
register_func(Softplus, 'softplus')
register_func(Arctang, 'arctg')

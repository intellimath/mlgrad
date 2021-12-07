# coding: utf-8

#from mlgrad.func cimport Func

cdef class Average_FG(Average):
    #
    def __init__(self, Penalty penalty, tol=1.0e-5, h=0.1, 
                 n_iter=1000, m_iter=20):
        """
        """
        self.penalty = penalty
        self.tol = tol
        self.n_iter = n_iter
        self.m_iter = m_iter
        self.h = h
        self.m = 0
    #
    def use_deriv_averager(self, averager):
        self.deriv_averager = averager
    #
    cdef init(self, float[::1] Y, u0=None):
        if self.deriv_averager is not None:
            self.deriv_averager.init()
        Average.init(self, Y, u0)

    #
    cdef fit_epoch(self, float[::1] Y):
        cdef float g
                        
        g = self.penalty.derivative(Y, self.u)
                
        if self.deriv_averager is not None:
            g = self.deriv_averager.update(g)

        self.u -= self.h * g
    #
#     cdef fit_epoch_s(self, float[::1] Y):
#         cdef float grad

#         grad = self.penalty.derivative_s(Y, self.u, self.s)
        
#         if self.deriv_averager is not None:
#             grad = self.deriv_averager.update(grad)

#         self.s -= self.h * grad
    #

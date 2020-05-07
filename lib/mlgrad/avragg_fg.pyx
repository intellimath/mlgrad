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
    cdef fit_epoch(self, double[::1] Y):
        cdef double g
                        
        g = self.penalty.derivative(Y, self.u)
                
        if self.deriv_averager is not None:
            g = self.deriv_averager.update(g)

        self.u -= self.h * g
    #
#     cdef fit_epoch_s(self, double[::1] Y):
#         cdef double grad

#         grad = self.penalty.derivative_s(Y, self.u, self.s)
        
#         if self.deriv_averager is not None:
#             grad = self.deriv_averager.update(grad)

#         self.s -= self.h * grad
    #

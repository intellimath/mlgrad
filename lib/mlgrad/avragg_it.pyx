#from mlgrad.func cimport Func

cdef class Average_Iterative(Average):
    #
    def __init__(self, Penalty penalty, tol=1.0e-6, n_iter=1000, m_iter=20):
        """
        """
        self.penalty = penalty
        self.tol = tol
        self.n_iter = n_iter
        self.m_iter = m_iter
        self.deriv_averager = None
        
        self.u = 0
        self.first = 1
    #
    cdef fit_epoch(self, double[::1] Y):
        self.u = self.penalty.iterative_next(Y, self.u)

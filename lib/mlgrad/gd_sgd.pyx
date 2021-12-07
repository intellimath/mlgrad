# coding: utf-8

cdef class SGD(GD):
    #
    def __init__(self, Risk risk, tol=1.0e-8, h=0.001, n_iter=1000,
                 stop_condition=None, callback=None, h_rate=None):
        """
        """
        self.risk = risk
        self.stop_condition = get_stop_condition(stop_condition)(self)
        self.grad_averager = None
        #self.param_averager = None
        self.tol = tol
        self.n_iter = n_iter
        self.callback = callback
        self.h = h
        self.param_prev = None

        if h_rate is None:
            self.h_rate = PowerParamRate(h, 0.67)
        else:
            self.h_rate = h_rate
    #
    cpdef fit_epoch(self):
        cdef Functional _risk = self.risk
        
        cdef int N = self.risk.n_sample
        cdef int n_param = len(self.risk.param)
        
        cdef int i, j, k, K
        
        cdef float[::1] grad_average
        
        cdef float h
        
        self.risk.lval = 0
        K = (self.K-1) * N + 1
        for j in range(N):
            # очередной случайный индекс
            k = rand(N)
                        
            h = self.h_rate.get_rate() #(K + j)

            _risk.gradient_loss(k)
            
            self.grad_averager.update(_risk.grad, h)
            grad_average = self.grad_averager.array_average
            for i in range(n_param):
                _risk.param[i] -= grad_average[i]                                



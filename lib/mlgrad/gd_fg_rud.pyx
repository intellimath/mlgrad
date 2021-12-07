# coding: utf-8

cdef class FG_RUD(GD):
    def __init__(self, Functional risk, tol=1.0e-8, h=0.001, n_iter=1000,
                 callback=None, stop_condition=None, h_rate=None, gamma = 1):
        """
        """
        self.risk = risk
        self.stop_condition = get_stop_condition(stop_condition)(self)
        self.grad_averager = None
        #self.param_averager = None
        self.tol = tol
        self.n_iter = n_iter
        self.h = h
        self.param_prev = None
        self.gamma = gamma
        
        if h_rate is None:
            self.h_rate = ConstantParamRate(h)
        else:
            self.h_rate = h_rate
            
        self.m = 0
        
        self.callback = callback
    #
    cpdef init(self):
        GD.init(self)
        n_param = len(self.risk.param)
        self.param_prev = np.zeros(n_param, 'f')
    #
    cpdef gradient(self):
        cdef Risk risk = self.risk
        cdef Py_ssize_t i, n_param = risk.param.shape[0]
        cdef float[::1] param = risk.param
        cdef float[::1] param_prev = self.param_prev
        cdef float[::1] array_average

        copy_memoryview(param_prev, param)

        array_average = self.grad_averager.array_average
        for i in range(n_param):
            param[i] -= array_average[i]
        self.risk.gradient()

        copy_memoryview(param, param_prev)
    #


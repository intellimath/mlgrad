# coding: utf-8

cdef class FG(GD):
    #
    def __init__(self, Functional risk, tol=1.0e-8, h=0.001, n_iter=1000, M = 12,
                 callback=None, stop_condition=None, h_rate=None):
        """
        """
        self.risk = risk
        self.stop_condition = get_stop_condition(stop_condition)(self)
        self.grad_averager = None
        #self.param_averager = None
        self.tol = tol
        self.n_iter = n_iter
        self.h = h
#         self.param_prev = None
#         self.gamma = gamma
#         self.normalizer = normalizer
        
        if h_rate is None:
#             self.h_rate = ExponentParamRate(h)
            self.h_rate = ConstantParamRate(h)
        else:
            self.h_rate = h_rate
            
        # self.m = 0
        self.M = M
        
        self.param_min = None
        
        self.callback = callback
    #
#     cpdef gradient(self):
#         self.risk.gradient()
#     #


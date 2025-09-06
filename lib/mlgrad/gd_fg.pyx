# coding: utf-8

cdef class FG(GD):
    #
    def __init__(self, Functional risk, tol=1.0e-9, h=0.001, n_iter=1000, M = 7,
                 callback=None, h_rate=None):
        """
        """
        self.risk = risk
        # self.stop_condition = get_stop_condition(stop_condition)(self)
        self.grad_averager = None
        # self.param_transformer = None
        self.tol = tol
        self.n_iter = n_iter
        self.h = h
#         self.param_prev = None
#         self.gamma = gamma
        self.normalizer = None

        if h_rate is None:
#             self.h_rate = ExponentParamRate(h)
            self.h_rate = ConstantParamRate(h)
        else:
            self.h_rate = h_rate

        self.M = M

        self.param_min = None
        self.param_copy = None

        self.callback = callback
    #
#     def gradient(self):
#         self.risk.gradient()
#     #


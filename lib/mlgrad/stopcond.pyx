
_stop_condition_dict = {
    'diffL1': DiffL1StopCondition,
    'diffL2': DiffL2StopCondition,
    'diffG1': DiffG1StopCondition,
    'diffP1': DiffP1StopCondition,
}

cpdef StopCondition get_stop_condition(label):
    return <StopCondition>_stop_condition_dict.get(label, DiffL1StopCondition)

cdef class StopCondition:
    #
    cdef init(self):
        pass
    #
    cdef bint verify(self):
        return 1
    #
#     cdef void finalize(self):
#         pass

cdef class DiffL1StopCondition(StopCondition):
    #
    def __init__(self, GD gd):
        self.gd = gd
    #
    cdef bint verify(self):
        cdef GD gd = self.gd
        
        if gd.K <= 2:
            return 0

        if fabs(gd.lval - gd.lval_min) / (1.0 + fabs(gd.lval_min)) < gd.tol:
            return 1
        
        return 0
    #            
        
cdef class DiffL2StopCondition(StopCondition):
    #
    def __init__(self, GD gd):
        self.gd = gd
    #
    cdef init(self):
        self.lval1 = self.lval2 = 0
    #
    cdef bint verify(self):
        cdef GD gd = self.gd
        cdef Functional risk = gd.risk
        cdef double lval_min

        #self.lval = risk.evaluate()

        if gd.K <= 3:
            self.lval1, self.lval2 = gd.lval, self.lval1
            return 0

        lval_min = min3(gd.lval, self.lval1, self.lval2)

        if 0.5 * fabs(gd.lval - 2*self.lval1 + self.lval2) / (1.0 + fabs(lval_min)) < gd.tol:
            return 1

        self.lval1, self.lval2 = gd.lval, self.lval1
            
        return 0
    #
#     cdef void finalize(self):
#         cdef GD gd = self.gd
#         cdef Functional risk = gd.risk
#         cdef int i, m = len(risk.param)
        
#         for i in range(m):
#             risk.param[i] = 0.5 * (risk.param[i] + gd.param_prev[i])


cdef class DiffG1StopCondition(StopCondition):
    #
    def __init__(self, GD gd):
        self.gd = gd
    #
    cdef init(self):
        m = len(self.gd.risk.param)
        self.grad = np.zeros((m,), "d")
    #
    cdef bint verify(self):
        cdef GD gd = self.gd
        cdef double[::1] grad_average = gd.risk.grad_average
        cdef double dg
        cdef int i, m = len(grad_average)

        if gd.K <= 2:
            copy_memoryview(self.grad, grad_average)
            return 0

        dg = 0
        for i in range(m):
            dg += fabs(self.grad[i] - grad_average[i])
        copy_memoryview(self.grad, grad_average)

        if dg < gd.tol:
            return 1

        return 0

cdef class DiffP1StopCondition(StopCondition):
    #
    def __init__(self, GD gd):
        self.gd = gd
    #
    cdef init(self):
        m = len(self.gd.risk.param)
        self.param = np.zeros((m,), "d")
    #
    cdef bint verify(self):
        cdef GD gd = self.gd
        cdef double[::1] param = gd.risk.param
        cdef double dp
        cdef int i, m = len(param)

        if gd.K <= 2:
            copy_memoryview(self.param, param)
            return 0

        dp = 0
        for i in range(m):
            dp += fabs(self.param[i] - param[i])
        copy_memoryview(self.param, param)

        if dp < gd.tol:
            return 1

        return 0

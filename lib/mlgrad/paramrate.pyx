
cdef class ParamRate:

    cpdef init(self):
        self.K = 1
                
    cdef float get_rate(self):
        return 1.0
        
cdef class ConstantParamRate(ParamRate):
        
    def __init__(self, h):
        self.h = h
        
    cdef float get_rate(self):
        self.K += 1
        return self.h

cdef class ExponentParamRate(ParamRate):

    def __init__(self, h, p=0.999):
        self.h = h
        self.curr_h = h
        self.p = p
        
    cpdef init(self):
        self.curr_h = self.h
        self.K = 1

    cdef float get_rate(self):
        cdef float h = self.curr_h
        self.curr_h *= self.p
        self.K += 1
        return h

cdef class PowerParamRate(ParamRate):

    def __init__(self, h, p=0.67):
        self.h = h
        self.p = p

    cdef float get_rate(self):
        cdef float t = self.K
        cdef float h = self.h / pow(t, self.p)
        self.K += 1
        return h


cpdef ParamRate get_learn_rate(key, args):
    if key == 'const':
        return ConstantParamRate(*args)
    elif key == 'exp':
        return ExponentParamRate(*args)
    elif key == 'pow':
        return PowerParamRate(*args)
    else:
        raise KeyError('Invalid learning rate name')
        
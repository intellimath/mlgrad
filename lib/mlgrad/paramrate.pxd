
cdef class ParamRate:
    cdef public double h
    cdef public int K

    cpdef init(self)
    cdef double get_rate(self)
    
cdef class ConstantParamRate(ParamRate):
    pass
    
cdef class ExponentParamRate(ParamRate):
    cdef public double curr_h
    cdef public double p

cdef class PowerParamRate(ParamRate):
    cdef public double p

cpdef ParamRate get_learn_rate(key, args)

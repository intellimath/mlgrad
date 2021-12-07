
cdef class ParamRate:
    cdef public float h
    cdef public int K

    cpdef init(self)
    cdef float get_rate(self)
    
cdef class ConstantParamRate(ParamRate):
    pass
    
cdef class ExponentParamRate(ParamRate):
    cdef public float curr_h
    cdef public float p

cdef class PowerParamRate(ParamRate):
    cdef public float p

cpdef ParamRate get_learn_rate(key, args)

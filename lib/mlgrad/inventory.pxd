
cdef void fa_fill(float *to, const float c, const size_t n) nogil
cdef float fa_conv(const float*, const float*, const size_t) nogil
cdef void fa_move(float*, const float*, const size_t) nogil
cdef float fa_sum(const float*, const size_t) nogil
cdef void fa_mul_add_array(float *a, const float *b, float c, const size_t n) nogil
cdef void fa_mul_const(float *a, const float c, const size_t n) nogil
cdef void fa_matdot(float*, float*, const float*, const size_t, const size_t) nogil
cdef void fa_matdot2(float*, float*, const float*, const size_t, const size_t) nogil
cdef void fa_mul_add_arrays(float *a, const float *b, const float *ss, const size_t n_input, const size_t n_output) nogil
cdef void fa_mul_grad(float *grad, const float *X, const float *ss, const size_t n_input, const size_t n_output) nogil

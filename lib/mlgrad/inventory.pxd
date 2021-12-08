
cdef void fa_fill(double *to, const double c, const size_t n) nogil
cdef double fa_conv(const double*, const double*, const size_t) nogil
cdef void fa_move(double*, const double*, const size_t) nogil
cdef double fa_sum(const double*, const size_t) nogil
cdef void fa_mul_add_array(double *a, const double *b, double c, const size_t n) nogil
cdef void fa_mul_const(double *a, const double c, const size_t n) nogil
# cdef void fa_matdot(double*, double*, const double*, const size_t, const size_t) nogil
cdef void fa_matdot2(double*, double*, const double*, const size_t, const size_t) nogil
cdef void fa_mul_add_arrays(double *a, const double *b, const double *ss, const size_t n_input, const size_t n_output) nogil
cdef void fa_mul_grad(double *grad, const double *X, const double *ss, const size_t n_input, const size_t n_output) nogil

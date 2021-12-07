cdef void fa_fill(float *to, const float c, const size_t n) nogil:
    cdef size_t i
    for i in range(n):
        to[i] = c

cdef void fa_move(float *to, const float *src, const size_t n) nogil:
    cdef size_t i
    for i in range(n):
        to[i] = src[i]

cdef float fa_conv(const float *a, const float *b, const size_t n) nogil:
    cdef size_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[i]
    return s

cdef float fa_sum(const float *a, const size_t n) nogil:
    cdef size_t i
    cdef double s = 0

    for i in range(n):
        s += a[i]
    return s

cdef void fa_mul_const(float *a, const float c, const size_t n) nogil:
    cdef size_t i

    for i in range(n):
        a[i] *= c

cdef void fa_mul_add_array(float *a, const float *b, float c, const size_t n) nogil:
    cdef size_t i
    
    for i in range(n):
        a[i] += c * b[i]

# cdef void fa_matdot(float *output, float *M, const float *X, 
#                     const size_t n_input, const size_t n_output) nogil:
#     cdef size_t i, j
#     cdef double s
#     cdef float *Mj = M

#     for j in range(n_output):
#         s = 0
#         for i in range(n_input):
#             s += Mj[i] * X[i];
#         output[j] = s
#         Mj += n_input

cdef void fa_matdot2(float *output, float *M, const float *X, 
                    const size_t n_input, const size_t n_output) nogil:
    cdef size_t i, j
    cdef double s
    cdef float *Mj = M;

    for j in range(n_output):
        s = Mj[0]
        Mj += 1
        for i in range(n_input):
            s += Mj[i] * X[i]
        output[j] = s
        Mj += n_input

cdef void fa_mul_add_arrays(float *a, float *M, const float *ss, const size_t n_input, const size_t n_output) nogil:
    cdef size_t i, j
    cdef float *Mj = M;
    cdef double sx

    for j in range(n_output):
        Mj += 1
        sx = ss[j]
        for i in range(n_input):
            a[i] += sx * Mj[i]
        Mj += n_input

cdef void fa_mul_grad(float *grad, const float *X, const float *ss, const size_t n_input, const size_t n_output) nogil:
    cdef size_t i, j
    cdef float *G = grad
    cdef double sx
    
    for j in range(n_output):
        sx = ss[j]
        G[0] = sx
        G += 1
        for i in range(n_input):
            G[i] = sx * X[i]
        G += n_input

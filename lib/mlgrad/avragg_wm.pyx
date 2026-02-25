@cython.final
cdef class WMAverage(Average):
    #
    def __init__(self, Average avr):
        self.avr = avr
        self.u = 0
        self.evaluated = 0
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _evaluate(self, double[::1] Y):
        cdef double v, yk, avr_u
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S

        avr_u = self.avr._evaluate(Y)

        S = 0
        # for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        for k in range(N):
            yk = Y[k]
            v = yk if yk <= avr_u else avr_u
            S += v
        self.u = S / N

        # self.u_min = self.u
        self.K = self.avr.K
        self.evaluated = 1
        return self.u
    #
    @cython.cdivision(True)
    @cython.final
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double u, v, m, fN = N

        if self.evaluated == 0:
            u = self.avr._evaluate(Y)
        else:
            u = self.avr.u
        self.avr._gradient(Y, grad)
        self.evaluated = 0

        m = 0
        for k in range(N):
            if Y[k] > u:
                m += 1

        # for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        for k in range(N):
            v = m * grad[k]
            if Y[k] <= u:
                v = v + 1
            grad[k] = v / fN
    #
    @cython.cdivision(True)
    @cython.final
    cdef _derivative_div(self, double[::1] Y, double[::1] G):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double u, v, m, fN = N

        if self.evaluated == 0:
            u = self.avr._evaluate(Y)
        else:
            u = self.avr.u
        self.avr._derivative_div(Y, G)
        self.evaluated = 0

        m = 0
        for k in range(N):
            if Y[k] > u:
                m += 1

        # for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        for k in range(N):
            v = m * G[k]
            if Y[k] <= u:
                v = v + 1
            G[k] = v / fN
    #

# cdef class WMAverageMixed(Average):
#     #
#     def __init__(self, Average avr, double gamma=1):
#         self.avr = avr
#         self.gamma = gamma
#         self.u = 0
#         self.evaluated = 0
#     #
#     cpdef fit(self, double[::1] Y):
#         cdef double u, v, yk, avr_u
#         cdef Py_ssize_t k, m, N = Y.shape[0]

#         self.avr.fit(Y)
#         self.evaluated = 1
#         avr_u = self.avr.u

#         m = 0
#         for k in range(N):
#             if Y[k] > avr_u:
#                 m += 1

#         u = 0
#         v = 0
#         # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
#         for k in range(N):
#             yk = Y[k]
#             if yk <= avr_u:
#                 u += yk
#             else:
#                 v += yk

#         self.u = (1-self.gamma) * u / (N-m) + self.gamma * v / m

#         self.u_min = self.u
#     #
#     cdef _gradient(self, double[::1] Y, double[::1] grad):
#         cdef Py_ssize_t k, m, N = Y.shape[0]
#         cdef double v, N1, N2, yk, avr_u

#         self.avr._gradient(Y, grad)
#         self.evaluated = 0
#         avr_u = self.avr.u

#         m = 0
#         for k in range(N):
#             if Y[k] > avr_u:
#                 m += 1

#         N1 = (1-self.gamma) / (N-m)
#         N2 = self.gamma / m
#         # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
#         for k in range(N):
#             yk = Y[k]
#             if yk <= avr_u:
#                 v = N1
#             else:
#                 v = N2
#             grad[k] = v
#     #

@cython.final
cdef class WMZAverage(Average):
    #
    def __init__(self, MAverage mavr=None, MAverage savr=None, func=SoftAbs_Sqrt(0.001), c=1.0/0.6745, alpha=3.5):
        self.func = func
        if mavr is None:
            self.mavr = MAverage(func)
        else:
            self.mavr = mavr
        if savr is None:
            self.savr = MAverage(func)
        else:
            self.savr = savr
        self.c = c
        self.alpha = alpha * c
        self.U = None
        self.GU = None
        self.evaluated = 0
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _evaluate(self, double[::1] Y):
        cdef Py_ssize_t j, N = Y.shape[0]
        cdef double[::1] U = self.U
        cdef Func rho_func = self.savr.func
        cdef double mval, tval, v, s

        self.mval = self.mavr._evaluate(Y)

        if U is None or U.shape[0] != N:
            U = self.U = inventory.empty_array(N)

        mval = self.mval
        for j in range(N):
            U[j] = rho_func._evaluate(Y[j] - mval)

        self.sval = rho_func._inverse(self.savr._evaluate(U))
        tval = self.mval + self.alpha * self.sval

        s = 0
        for j in range(N):
            v = Y[j]
            if v > tval:
                v = tval
            s += v
        s /= N
        self.u = s
        self.evaluated = 1

        return s
    #
    @cython.cdivision(True)
    @cython.final
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t j, N = Y.shape[0]
        cdef double[::1] GU = self.GU
        cdef Func rho_func = self.savr.func

        cdef double mval, tval, alpha, v, ss, m

        if not self.evaluated:
            self._evaluate(Y)

        if GU is None or GU.shape[0] != N:
            GU = self.GU = inventory.empty_array(N)

        mval = self.mval
        alpha = self.alpha
        tval = mval + alpha * self.sval

        m = 0
        for j in range(N):
            if Y[j] >= tval:
                m += 1

        if m > 0:
            self.mavr._gradient(Y, grad)
            self.savr._gradient(self.U, GU)

            for j in range(N):
                GU[j] *= rho_func._derivative(Y[j] - mval)

            ss = 0
            for j in range(N):
                ss += GU[j]

            v = rho_func._derivative(self.sval)
            if v == 0:
                for j in range(N):
                    grad[j] = 0
            else:
                for j in range(N):
                    grad[j] = m * (grad[j] + alpha * (GU[j] - ss * grad[j]) / v)

            for j in range(N):
                if Y[j] < tval:
                    grad[j] += 1

            for j in range(N):
                grad[j] /= N
        else:
            inventory.fill(grad, 1.0/N)

        self.evaluated = 0

@cython.final
cdef class WMZSum(Average):
    #
    def __init__(self, MAverage mavr=None, MAverage savr=None, c=1.0/0.6745, alpha=3.5):
        cdef Func func = SoftAbs_Sqrt(0.001)
        if mavr is None:
            self.mavr = MAverage(func)
        else:
            self.mavr = mavr
        if savr is None:
            self.savr = MAverage(func)
        else:
            self.savr = savr
        self.c = c
        self.alpha = alpha * c
        self.U = None
        self.GU = None
        self.evaluated = 0
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _evaluate(self, double[::1] Y):
        cdef Py_ssize_t j, N = Y.shape[0]
        cdef double[::1] U = self.U
        cdef Func rho_func = self.savr.func
        cdef double mval, tval, v, s

        self.mval = self.mavr._evaluate(Y)

        if U is None or U.shape[0] != N:
            U = self.U = inventory.empty_array(N)

        mval = self.mval
        for j in range(N):
            U[j] = rho_func._evaluate(Y[j] - mval)

        self.sval = rho_func._inverse(self.savr._evaluate(U))
        tval = self.mval + self.alpha * self.sval

        s = 0
        for j in range(N):
            v = Y[j]
            if v >= tval:
                v = tval
            s += v
        self.u = s
        self.evaluated = 1
    
        return s
    #
    @cython.cdivision(True)
    @cython.final
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t j, N = Y.shape[0]
        cdef double[::1] GU = self.GU
        cdef Func rho_func = self.mavr.func
        cdef double mval, tval, alpha, v, ss, m

        if not self.evaluated:
            self._evaluate(Y)

        if GU is None or GU.shape[0] != N:
            GU = self.GU = inventory.empty_array(N)

        mval = self.mval
        alpha = self.alpha
        tval = mval + alpha * self.sval

        m = 0
        for j in range(N):
            if Y[j] >= tval:
                m += 1

        if m > 0:
            self.mavr._gradient(Y, grad)
            self.savr._gradient(self.U, GU)

            for j in range(N):
                GU[j] *= rho_func._derivative(Y[j] - mval)

            ss = 0
            for j in range(N):
                ss += GU[j]
            # print(ss, end=' ')

            v = rho_func._derivative(self.sval)
            if v == 0:
                for j in range(N):
                    grad[j] = 0
            else:
                for j in range(N):
                    grad[j] = m * (grad[j] + alpha * (GU[j] - ss * grad[j]) / v)

            for j in range(N):
                if Y[j] < tval:
                    grad[j] += 1

            # for j in range(N):
            #     grad[j] /= N
        else:
            inventory.fill(grad, 1.0)

        self.evaluated = 0


cdef class WZAverage(Average):
    #
    def __init__(self, alpha=3.0):
        self.alpha = alpha
        self.evaluated = 0
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _evaluate(self, double[::1] Y):
        cdef Py_ssize_t j, N = Y.shape[0]
        cdef double tval1, tval2, v, s

        self.mval = inventory._mean(&Y[0], N)
        self.sval = inventory._std(&Y[0], self.mval, N)
        tval1 = self.mval + self.alpha * self.sval
        tval2 = self.mval - self.alpha * self.sval

        s = 0
        for j in range(N):
            v = Y[j]
            if v >= tval1:
                v = tval1
            elif v <= tval2:
                v = tval2
            s += v
        s /= N
        self.u = s
        self.evaluated = 1

        return s
    #
    @cython.cdivision(True)
    @cython.final
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t j, N = Y.shape[0]
        cdef double mval=self.mval, sval=self.sval, tval1, tval2
        cdef double alpha=self.alpha, v, m1, m2, g

        if not self.evaluated:
            self._evaluate(Y)

        tval1 = mval + alpha * sval
        tval2 = mval - alpha * sval

        m1 = m2 = 0
        for j in range(N):
            v = Y[j]
            if v >= tval1:
                m1 += 1
            elif v <= tval2:
                m2 += 1

        for j in range(N):
            v = Y[j]
            g = 1
            if v >= tval1:
                g += m1*(1 + alpha)
            elif v <= tval2:
                g += m2*(1 - alpha)
            grad[j] = g / N

        self.evaluated = 0
    #


cdef class ERiskGB(Risk):
    #
    def __init__(self, double[:,::1] X not None, double[::1] Y not None, Model model not None,
                 Loss loss=None, Batch batch=None,
                 alpha=1.0, is_natgrad=0):

        self.model = model
        self.param = model.param

        self.n_param = model.n_param
        self.n_input = model.n_input
#         if self.model.grad is None:
#             self.model.grad = np_zeros(self.n_param, np_double)

#         if self.model.grad_input is None:
#             self.model.grad_input = np_zeros(self.n_input, np_double)

        if loss is None:
            self.loss = ErrorLoss(Square())
        else:
            self.loss = loss

        if self.model._is_regularized():
            self.grad_r = np_zeros(self.n_param, np_double)

        self.grad = np.zeros(self.n_param, np_double)
        self.grad_average = np_zeros(self.n_param, np_double)

        self.X = X
        self.Y = Y
        self.n_sample = len(Y)

        if batch is None:
            self.use_batch(WholeBatch(self.n_sample))
        else:
            self.use_batch(batch)

        N = len(Y)
        self.weights = np.full(N, 1./N, np_double)
        self.lval = 0

        self.H = np_zeros(self.n_sample, np_double)
        self.alpha = alpha
        self.is_natgrad = is_natgrad
    #
    def use_weights(self, weights not None):
        self.weights = weights
    #
    # cdef void _evaluate_models_all(self, double[::1] vals):
    #     cdef Py_ssize_t k
    #     cdef Model _model = self.model
    #     cdef Loss _loss = self.loss

    #     cdef double[:, ::1] X = self.X
    #     cdef double alpha = self.alpha
    #     cdef double[::1] H = self.H

    #     for k in range(self.n_sample):
    #         vals[k] = H[k] + alpha * _model._evaluate_one(X[k])
    #
#     cdef void _evaluate_models_batch(self):
#         cdef Py_ssize_t j, k
#         cdef double y
#         cdef Model _model = self.model
#         cdef Loss _loss = self.loss

#         cdef double[:, ::1] X = self.X
#         cdef double[::1] Y = self.Y
#         cdef Py_ssize_t[::1] indices = self.batch.indices
#         cdef double alpha = self.alpha
#         cdef double[::1] H = self.H
#         cdef double[::1] model_vals = self.model_vals

#         for j in range(self.batch.size):
#             k = indices[j]
#             model_vals[j] = H[k] + alpha * _model._evaluate_one(X[k])
    #
    cdef void _evaluate_losses_batch(self):
        cdef Py_ssize_t j, k
        cdef double y
        cdef Loss _loss = self.loss
        cdef Model _model = self.model
        cdef double alpha = self.alpha

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] model_vals = self.model_vals
        cdef double[::1] L = self.loss_vals
        cdef double[::1] H = self.H

        for j in range(self.batch.size):
            k = indices[j]
            y = model_vals[j] = H[k] + alpha * _model._evaluate_one(X[k])
            L[j] = _loss._evaluate(y, Y[k])
    #
    cdef void _evaluate_losses_all(self, double[::1] lvals):
        cdef Py_ssize_t j, k
        cdef double yk
        cdef Loss _loss = self.loss
        cdef Model _model = self.model
        cdef double alpha = self.alpha

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        # cdef double[::1] L = self.loss_vals
        cdef double[::1] H = self.H
        # cdef Py_ssize_t N = X.shape[0]

        for k in range(self.n_sample):
            yk = H[k] + alpha * _model._evaluate_one(X[k])
            lvals[k] = _loss._evaluate(yk, Y[k])
    #
#     cdef void _evaluate_losses_derivative_div_batch(self):
#         cdef Py_ssize_t j, k
#         cdef double y
#         cdef Loss _loss = self.loss
#         cdef Model _model = self.model

#         cdef double[:, ::1] X = self.X
#         cdef double[::1] Y = self.Y
#         cdef Py_ssize_t[::1] indices = self.batch.indices
#         cdef double alpha = self.alpha

#         # cdef double[::1] model_vals = self.model_vals
#         cdef double[::1] LD = self.loss_valsD
#         cdef double[::1] H = self.H

#         for j in range(self.batch.size):
#             k = indices[j]
#             y = H[k] + alpha * _model._evaluate_one(X[k])
#             LD[j] = _loss._derivative_div(y, Y[k])
    #
    cdef void _evaluate_losses_derivative_div_all(self, double[::1] lvals):
        cdef Py_ssize_t k
        cdef double y
        cdef Loss _loss = self.loss
        cdef Model _model = self.model

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double alpha = self.alpha

        # cdef double[::1] model_vals = self.model_vals
        # cdef double[::1] LD = self.loss_valsD
        cdef double[::1] H = self.H

        for k in range(self.n_sample):
            y = H[k] + alpha * _model._evaluate_one(X[k])
            lvals[k] = _loss._derivative_div(y, Y[k])
    #
    cdef double _evaluate(self):
        cdef Py_ssize_t j, k
        cdef double S, y
        cdef Py_ssize_t[::1] indices = self.batch.indices

        # cdef Model _model = self.model

        cdef double[::1] weights = self.weights

        cdef double[::1] L = self.loss_vals

        # self._evaluate_models_batch()
        self._evaluate_losses_batch()

        S = 0
        for j in range(self.batch.size):
            k = indices[j]
            S += weights[k] * L[j]

        if self.model._is_regularized():
            S += self.model._evaluate_reg()

        self.lval = S
        return S
    #
    cdef void _gradient(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t i, j, k
        cdef double y, vv

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t[::1] indices = self.batch.indices

        cdef double alpha = self.alpha
        cdef double[::1] model_vals = self.model_vals
        # cdef double[::1] L = self.loss_vals
        cdef double[::1] H = self.H

        inventory.clear(self.grad_average)

        # self._evaluate_models_batch()

        for j in range(self.batch.size):
            k = indices[j]

            # y = H[j] + alpha * _model._evaluate_one(X[k])
            _model._gradient_one(X[k], grad)

            vv = alpha * _loss._derivative(model_vals[j], Y[k]) * weights[k]
            for i in range(self.n_param):
                self.grad_average[i] += vv * grad[i]

        if self.model._is_regularized():
            self.add_regularized_gradient()
    #
    cdef double derivative_alpha(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t j, k, N = self.n_sample
        cdef double y, v

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double alpha = self.alpha
        cdef double[::1] H = self.H
        cdef double ret = 0

        cdef double[::1] model_vals = self.model_vals
        cdef double[::1] L = self.loss_vals

        for j in range(size):
            k = indices[j]

            v = _model._evaluate_one(X[k])
            y = H[k] + alpha * v
            ret += _loss._derivative(y, Y[k]) * weights[k] * v

        return ret

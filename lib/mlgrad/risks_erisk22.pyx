
cdef class ERisk22(Risk):
    #
    def __init__(self, double[:,::1] X, double[:,::1] Y,
                MLModel model, MultLoss2 loss,
                Batch batch=None, is_natgrad=0):
        self.model = model
        self.param = model.param
        self.loss = loss
        # self.regnorm = regnorm
        self.weights = None
        self.grad = None
        self.grad_u = None
        self.grad_r = None
        self.grad_average = None
        self.X = X
        self.Y = Y
        self.n_sample = len(Y)
        if batch is None:
            self.batch = WholeBatch(self.n_sample)
        else:
            self.batch = batch

        self.loss_vals = np.zeros(self.batch.size, 'd')
        self.is_natgrad = is_natgrad
        #
    #
    def use_weights(self, weights):
        self.weights = weights
    #
    #cdef object get_loss(self):
    #    return self.loss
    #
    cpdef init(self):
        N = self.n_sample
        self.n_param = self.model.n_param
        self.n_input = self.model.n_input
        self.n_output = self.model.n_output

        # if self.model.grad is None:
        #     self.model.grad = np.zeros((n_param,), np_double)

        if self.grad is None:
            self.grad = np.zeros(self.n_param, dtype=np_double)

        if self.grad_u is None:
            self.grad_u = np.zeros(self.n_output, dtype=np_double)

        if self.grad_average is None:
            self.grad_average = np.zeros(self.n_param, dtype=np_double)

        if self.model._is_regularized():
            self.grad_r = np.zeros(self.n_param, dtype=np_double)

        if self.weights is None:
            self.weights = np.full((N,), 1./N, np_double)

        self.lval = 0
    #
    # cdef void generate_samples(self, X, Y):
    #     cdef double[:,::1] X1 = X
    #     cdef double[:,::1] Y1 = Y
    #     self.batch.generate_sample2d(Y1, self.Y)
    #     self.batch.generate_sample2d(X1, self.X)
    #
    cdef void _evaluate_losses_batch(self):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef MLModel _model = self.model
        cdef MultLoss2 _loss = self.loss
        #cdef double v
        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef double[::1] output = _model.output

        cdef Py_ssize_t size = self.batch.size
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] L = self.L

        for j in range(size):
            k = indices[j]
            _model._forward(X[k])
            L[k] = _loss._evaluate(output, Y[k])
    #
    cdef double _evaluate(self):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef double y, lval, S

        cdef MLModel _model = self.model
        cdef MultLoss2 _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef double[::1] output = _model.output
        cdef double[::1] weights = self.weights

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices

        S = 0
        for j in range(size):
            k = indices[j]
            _model._forward(X[k])
            # print(np.asarray(_model.output))
            lval = _loss._evaluate(_model.output, Y[k])
            S += weights[k] * lval

        if self.model._is_regularized():
            S += self.model._evaluate_reg()

        self.lval = S
        return S
    #
    cdef void _evaluate_losses_all(self, double[::1] lvals):
        cdef MLModel _model = self.model
        cdef MultLoss2 _loss = self.loss
        cdef double[::1] output = _model.output

        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef Py_ssize_t k
        cdef double[::1] yk

        for k in range(self.n_sample):
            _model._forward(X[k])
            lvals[k] = _loss._evaluate(output, Y[k])
    #
    cdef void _gradient(self):
        cdef Py_ssize_t j, k, n_param = self.model.n_param, N = self.n_sample
        cdef double y, yk, wk, S

        cdef MLModel _model = self.model
        cdef MultLoss2 _loss = self.loss
        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef double[::1] output = _model.output
        cdef double[::1] weights = self.weights
        cdef double[::1] Xk, Yk
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_u = self.grad_u
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t size = self.batch.size
        cdef Py_ssize_t[::1] indices = self.batch.indices

        inventory.fill(grad_average, 0)

        for j in range(size):
            k = indices[j]
            Xk = X[k]
            Yk = Y[k]

            _model._forward(Xk)
            _loss._gradient(output, Yk, grad_u)
            _model._backward(Xk, grad_u, grad)

            wk = weights[k]

            for i in range(n_param):
                grad_average[i] += wk * grad[i]

        inventory.move(self.grad, self.grad_average)

        if _model._with_eqns():
            self.add_equations_gradient()

        if _model.projection:
            self.project_equations()

        if _model._is_regularized():
            self.add_regularized_gradient()

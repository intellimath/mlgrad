def root_function_newton_raphson(f, df, x0, double h=1.0, double tol=1e-6, Py_ssize_t n_iter=1000):
    """
    Find root of f(x) = 0 using Newton-Raphson method.

    Args:
        f:      Function to find root of
        df:     Derivative of f
        x0:     Initial guess
        h:      Step
        tol:    Stop when |f(x)| < tol
        n_iter: Maximum iterations

    Returns:
        Approximate root
    """
    cdef Py_ssize_t i
    cdef double x, fx, dfx

    x = x0
    fx = f(x)

    for i in range(n_iter):
        if fabs(fx) > tol:
            break
        dfx = df(x)

        if dfx == 0:
            raise ValueError("Derivative is zero, method fails")

        x -= h * (fx / dfx)  # Newton-Raphson formula
        fx = f(x)

    return x
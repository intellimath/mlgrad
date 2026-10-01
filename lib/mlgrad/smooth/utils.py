import math
import numpy as np

# def whittaker_matrices(y, tau, W=None, W2=None, d=2):
#     N = len(y)
#     D = np.diff(np.eye(N), d, axis=0)
#     if W is None:
#         W = np.ones(N, "d")
#     WW = np.diag(W)
#     if W2 is None:
#         W2 = np.ones(N-d, "d")
#     WW2 = np.diag(W2)
#     Z = WW + tau * (D.T @ (WW2 @ D))
#     Y = WW @ y
#     return Z, Y

# cdef void diagonal_up(cdef double *x, Py_ssize_t n, Py_ssize_t d, double *y): 
#     pass

def create_banded(mat, d):
    """Create a banded matrix from a given quadratic Matrix.

    The Matrix will to be returned as a flattend matrix.
    Either in a column-wise flattend form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Then use::

      col_wise=True

    Or in a row-wise flattend form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    Then use::

      col_wise=False

    Dup1 and Dup2 or the first and second upper minor-diagonals and Dlow1 resp.
    Dlow2 are the lower ones. The number of upper and lower minor-diagonals can
    be altered.

    Parameters
    ----------
    mat : :class:`numpy.ndarray`
        The full (n x n) Matrix.
    up : :class:`int`
        The number of upper minor-diagonals. Default: 2
    low : :class:`int`
        The number of lower minor-diagonals. Default: 2
    col_wise : :class:`bool`, optional
        Use column-wise storage. If False, use row-wise storage.
        Default: ``True``

    Returns
    -------
    :class:`numpy.ndarray`
        Bandend matrix
    """
    # mat = np.asanyarray(mat, dtype="d")
    if mat.ndim != 2:
        msg = "create_banded: matrix has to be 2D"
        raise ValueError(msg)
    if mat.shape[0] != mat.shape[1]:
        msg = "create_banded: matrix has to be n x n"
        raise ValueError(msg)

    up  = d
    low = d
    col_wise = True

    size = mat.shape[0]
    mat_flat = np.zeros((2*d+1, size))
    mat_flat[up, :] = mat.diagonal()

    if col_wise:
        for i in range(up):
            mat_flat[i, (up - i) :] = mat.diagonal(up - i)
        for i in range(low):
            mat_flat[-i - 1, : -(low - i)] = mat.diagonal(-(low - i))
    else:
        for i in range(up):
            mat_flat[i, : -(up - i)] = mat.diagonal(up - i)
        for i in range(low):
            mat_flat[-i - 1, (low - i) :] = mat.diagonal(-(low - i))
    return mat_flat

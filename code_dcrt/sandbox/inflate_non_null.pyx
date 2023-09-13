import numpy as np

cimport numpy as np
cimport cython

np.import_array()


def _inflate_non_null(
    np.ndarray non_zero_index,
    long delta,
    long n_features):

    cdef list list_non_zero
    list_non_zero = non_zero_index.tolist()
    if n_features is None and delta > 0:
        raise ValueError(
            'When delta is positive n_features should be inputted')
    if n_features is None or delta == 0:
        return non_zero_index

    for s in non_zero_index:
        for i in range(delta):
            if s - i >= 0:
                list_non_zero.append(s - i)
            if s + i <= n_features - 1:
                list_non_zero.append(s + i)

    return np.unique(list_non_zero)

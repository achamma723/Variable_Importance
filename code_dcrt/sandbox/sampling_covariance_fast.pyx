import numpy as np

cimport cython
cimport numpy as np

np.import_array()


def sample_covariance_group(
    int n_vars,
    np.ndarray[long, ndim=1, mode='c'] labels,
    double rho,
    double gamma):
    """Follow method described in section 5 of Dai & Barber (2016)
    arXiv:1602.03589v1 to create covariance matrix with group structured"""
    
    cdef list Sigma = []

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                Sigma.append(1.0)
            elif labels[i] == labels[j]:
                Sigma.append(rho)
            else:
                Sigma.append(rho * gamma)
    
    return Sigma

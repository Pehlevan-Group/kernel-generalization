import numpy as np
import scipy as sp
from numba import jit, int64

###############################################################
################# Use Only These Functions ####################
###############################################################

@jit(nopython=True)
def gegenbauer(x, kmax, d):
    alpha = d / 2 - 1
    Q = np.zeros((kmax, len(x)))
    Q[0, :] = np.ones(len(x))
    Q[1, :] = 2 * alpha * x
    for k_m in range(kmax - 2):
        k = k_m + 2
        Q[k, :] = 1 / k * (2 * x * (k + alpha - 1) * Q[k - 1, :] - (k + 2 * alpha - 2) * Q[k - 2, :])
    return Q
    
def gegenbauer_gpu(x, kmax, d):
    import cupy as cp
    alpha = d / 2 - 1
    
    Q = cp.zeros((kmax, len(x)))
    Q[1, :] = 2 * alpha * x
    for k_m in range(kmax - 2):
        k = k_m + 2
        Q[k, :] = 1 / k * (2 * x * (k + alpha - 1) * Q[k - 1, :] - (k + 2 * alpha - 2) * Q[k - 2, :])
       
    del x
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return Q

def eigenvalue_normalization(kmax, alpha, degens):
    area_ratio = surface_area(2 * alpha + 2) / surface_area(2 * alpha + 1)

    n_factor = np.zeros(kmax)
    nbar_factor = np.zeros(kmax)

    for k in range(kmax):
        deg_k = degens[k]
        n_factor[k] = deg_k * area_ratio * (alpha / (alpha + k)) ** 2
        nbar_factor[k] = area_ratio * (alpha / (alpha + k))

    return nbar_factor
    "Mathematica definition: divide the integral result by it to get lambda_bar"


def degeneracy_kernel(dim, k):
    alpha = dim / 2.0 - 1
    return (k + alpha) / alpha * sp.special.comb(k + 2 * alpha - 1, k)
    "Mathematica definition"


def surface_area(d):
    return 2 * np.pi ** (d / 2) / sp.special.gamma(d / 2)

###############################################################
################# Use Only These Functions ####################
###############################################################

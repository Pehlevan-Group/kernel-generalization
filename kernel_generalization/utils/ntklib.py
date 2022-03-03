import numpy as np
import scipy.special as sp
import mpmath as mp
from numba import jit, int64, generated_jit
mp.mp.dps = 200; mp.mp.pretty = True

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

#############################################################
################# NTK Spectrum Functions ####################
#############################################################

def f_mpmath(phi, L):
    if L == 1:
        return mp.acos(1 / mp.pi * mp.sin(phi) + (1 - 1 / mp.pi * mp.acos(mp.cos(phi))) * mp.cos(phi))
    elif L == 0:
        return mp.acos(mp.cos(phi))
    else:
        return f_mpmath(phi, L - 1)

@jit(forceobj=True,parallel=True)
def NTK_mpmath(phi, L):
    if L == 1:
        ntk = mp.cos(f_mpmath(phi, 1)) +  (1 - phi / mp.pi) * mp.cos(phi)
        return ntk
    else:
        a = phi
        for i in range(L - 1):
            a = f_mpmath(a, 1)
            
        ntk = mp.cos(f_mpmath(a, 1)) + NTK_mpmath(phi, L - 1) * (1 - a / mp.pi)
        return ntk
    
## Implements omega(D)/omega(D-1) in the paper
def area_ratio(dim):
    return np.sqrt(np.pi)*np.exp(sp.gammaln((dim-1)/2) - sp.gammaln((dim)/2))
    
def eigenvalue_normalization(kmax, alpha):
    
    dim = 2.0*alpha + 2.0
    ## Implements omega(D)/omega(D-1) in the paper
    area_ratio = np.sqrt(np.pi)*np.exp(sp.gammaln((dim-1)/2) - sp.gammaln((dim)/2))
    
    norm_factor = np.zeros(kmax)
    for k in range(kmax):
        norm_factor[k] = area_ratio * (alpha / (alpha + k))

    return norm_factor
    "Mathematica definition: divide the integral result by it to get lambda_bar"
    
## Recursive gegenbauer polynomial definition    
def gegenbauer(x, kmax, dim):
    alpha = (dim - 2)/2.0
    Q = np.zeros((kmax, len(x)))
    Q[0, :] = np.ones(len(x))
    Q[1, :] = 2 * alpha * x
    for k_m in range(kmax - 2):
        k = k_m + 2
        Q[k, :] = 1 / k * (2 * x * (k + alpha - 1) * Q[k - 1, :] - (k + 2 * alpha - 2) * Q[k - 2, :])
    return Q

## Gegenbauer polynomial degeneracy   
def degeneracy_kernel(dim, k):
    alpha = (dim - 2)/2.0
    return (k + alpha) / alpha * sp.comb(k + 2 * alpha - 1, k)
    "Mathematica definition"

def mpmath_kernel_spectrum(layers, kmax, alpha, degens, parallel_cores = 0):
    
    global integral # Crucial for mpmath parallelization
    
    from multiprocessing import Pool
    
    normalization = eigenvalue_normalization(kmax, alpha)

    Q = lambda alpha, k,z: mp.gegenbauer(k, alpha, z, zeroprec=1000) # Gegenbauer polynomial
    mu = lambda alpha, z: (1-z**2)**(alpha-0.5)                      # Spherical measure
    kernel = lambda layer, z: NTK_mpmath(mp.acos(z),layer)           # Kernel function
    
    eta_bar = np.zeros((len(layers), kmax))
    
    for i in range(len(layers)):
        def integral(k):
            return mp.chop(mp.quad(lambda z: kernel(layers[i],z)*Q(alpha,k,z)*mu(alpha,z), [-1,1]))
    
        if parallel_cores == 0:
            eta_bar[i, :] = [integral(k) for k in range(kmax)]/normalization
        else:
            p = Pool(parallel_cores)
            eta_bar[i, :] = np.array(p.map(integral, range(kmax)))/normalization
            p.close()
            p.join()

    eta_bar = eta_bar*(eta_bar > 1e-40)
    eta = eta_bar/np.outer(len(layers), degens)

    return eta, eta_bar


## Given arrays of input dimension
def compute_ntk_spectrum(dim, deg, layer, num_cores):
    
    """
    Given the array of input dimensions, desired degrees and layers,
    it computes the ntk eigenvalues.
    
    Inputs:
    dim: an array of input dimensions
    deg: the array of gegenbauer polynomials to project NTK on. 
    layer: the array of number of layers for NTK's. Bias is ignored and weight std = 1.
    num_cores: number of parallel cores
    
    Outputs:
    eig: Scaled eigenvalues with degeneracy
    eig_real: Unscaled eigenvalues (exact integral)
    
    Example:
    
    dim = np.array([5*(i+1) for i in range(2)]) # For input dimensions: [5, 10]
    deg = np.array([i for i in range(10)]);     # For degrees from Q_0 to Q_10
    layer = np.array([1, 2]);                   # For NTK layers [1, 2]

    eig, eig_real = ntklib.compute_ntk_spectrum(dim, deg, layer, num_cores)
    np.savez('GegenbauerEigenvalues.npz', eig=eig, eig_real=eig_real)
    
    ## Load saved eigenvalues
    eig = np.load('GegenbauerEigenvalues.npz', allow_pickle=True)['eig']
    eig_real= np.load('GegenbauerEigenvalues.npz', allow_pickle=True)['eig_real']
    """
    
    eig = []   ## Scaled eigenvalues with degeneracy
    eig_real = []  ## Unscaled eigenvalues (exact integral)

    for dimension in dim:

        alpha = (dimension - 2.0)/2.0
        print('dim', dimension)
        kmax = len(deg)
        degens = np.array([degeneracy_kernel(dimension, k) for k in range(kmax)])

        start = datetime.now()
        spec, spec_norm = mpmath_kernel_spectrum(layer, kmax, alpha, degens, parallel_cores = num_cores)
        stop = datetime.now()
        print('Elapsed time: ' + time_diff(start,stop))

        eig += [spec_norm]
        eig_real += [spec]

    eig = np.array(eig)
    eig_real = np.array(eig_real)
    
    return eig, eig_real


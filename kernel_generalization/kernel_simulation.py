import numpy as np
import scipy as sp
import scipy.optimize
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt
import math

###############################################################
################# Use Only These Functions ####################
###############################################################

################# In Paper Format ####################

def noise(alpha):
    theta = (np.pi+np.arctan2(8*np.sqrt(alpha-1),(8+9*alpha*(3*alpha-4))))/3
    return 3*(alpha-1)*(3*alpha-1-2*np.sqrt(alpha)*np.sqrt(9*alpha-8)*np.cos(theta))
    
def kappa_fn(kappa, *args):
    (p, lamb, spectrum, degens) = args
    return kappa - lamb - kappa * np.sum(spectrum*degens/(p*spectrum + kappa))

def kappa_fn_derivative(kappa, *args):
    (p, lamb, spectrum, degens) = args
    return 1 - np.sum(spectrum*degens/(p*spectrum + kappa)) + kappa*np.sum(spectrum*degens/(p*spectrum + kappa)**2)

def solve_kappa(pvals, lamb, spectrum, degens = []):
    if len(degens) == 0:
      degens = np.ones(len(spectrum))
    
    sols = np.zeros(len(pvals))
    for i, p in enumerate(pvals):
        args = (p, lamb, spectrum, degens)
        sols[i] = sp.optimize.root_scalar(kappa_fn, x0=p*np.amax(spectrum), args = args, 
                                          fprime = kappa_fn_derivative, method = 'newton').root
    return sols

def gamma_fn(p, kappa, spectrum, degens = []):
    if len(degens) == 0:
        degens = np.ones(len(spectrum))
        
    return p * degens * spectrum**2 / (kappa + spectrum*p)**2


def mode_error_pure(p, kappa, spectrum, degens, lamb, noise_var, pure_mode = None, zero_mode = False, lambda_0 = True):
    gamma = gamma_fn(p, kappa, spectrum, degens)
    coeff = 1 / (1 - np.sum(gamma))
    noiseless_term = kappa**2 * gamma/p
    noise_term = noise_var * gamma
    
    if pure_mode != None:
        ## Specify w_\rho entries
        kmax = len(spectrum)
        noisless_pure = np.zeros(kmax)
        noisless_pure[pure_mode] = 1
    else:
        noisless_pure = 1
    
    mode_error = coeff * (noiseless_term*noisless_pure + noise_term)
    
    if zero_mode:
        mode_error[0] = kappa**2*spectrum[0]**2/(2*p*spectrum[0]+kappa)**2*(1+np.sum(gamma))/(1-np.sum(gamma))
    elif lambda_0:
        mode_error[0] = 0
    
    cum_gen_error =  mode_error + coeff* np.append(np.cumsum(spectrum[1:]**2*degens[1:]),[0]) * gamma
    
    return mode_error, cum_gen_error
   
def simulate_pure_gen_error(pvals, spectrum, degens, noise_var, pure_mode = None, lamb=1e-10, zero_mode = False, lambda_0 = True):
    kappa = solve_kappa(pvals, lamb, spectrum, degens)

    errs_tot = np.zeros((len(pvals), len(noise_var)))
    mode_errs = np.zeros((len(pvals), len(spectrum), len(noise_var)))
    cum_gen_errs = np.zeros((len(pvals), len(spectrum), len(noise_var)))
    
    for j in range(len(noise_var)):
        noise = noise_var[j]
        for i in range(len(pvals)):
            mode_errs[i, :, j], cum_gen_errs[i,:,j] = mode_error_pure(pvals[i], kappa[i], spectrum, degens, 
                                                                 lamb, noise, pure_mode = pure_mode,
                                                                 zero_mode = zero_mode, lambda_0 = lambda_0)
            errs_tot[i, j] = np.sum(mode_errs[i, :, j])

    return mode_errs, errs_tot, cum_gen_errs
    
    
def simulate_asymptotic(p_vals, spectrum, degens, noise_var, lamb=1e-10, mode = 1):
    
    errs_tot = np.zeros((len(p_vals), len(noise_var)))
    
    l = mode
    
    lambda_l = spectrum*degens
    
    alpha = p_vals/degens[l]
    alpha_l = (lamb + np.sum(lambda_l[l:]))/lambda_l[l]
    Eg_inf = np.sum(spectrum[l+1:]**2*degens[l+1])
    
    kappa_l = (alpha_l-alpha)/2 + np.sqrt((alpha+alpha_l)**2-4*alpha)/2
    gamma = alpha/(alpha+kappa_l)**2
    
    for i in range(len(noise_var)):
        sigma_l = (noise_var[i] + Eg_inf)/(spectrum[l]**2*degens[l])
        
        noiseless = kappa_l**2/(kappa_l+alpha)**2
        
        errs_tot[:,i] = spectrum[l]**2*degens[l]*(noiseless + sigma_l*gamma)/(1-gamma) + Eg_inf
    
    return errs_tot

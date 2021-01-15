import numpy as np
import math
import matplotlib.pyplot as plt
from kernel_generalization.utils import gegenbauer
import scipy as sp
import scipy.special
import scipy.optimize
 
from kernel_generalization.utils import neural_tangent_kernel as ntk

###############################################################
################# Use Only These Functions ####################
###############################################################

def f(phi, L):
    if L == 1:
        return np.arccos(1 / np.pi * np.sin(phi) + (1 - 1 / np.pi * np.arccos(np.cos(phi))) * np.cos(phi))
    elif L == 0:
        return np.arccos(np.cos(phi))
    else:
        return f(phi, L - 1)


def NTK(phi, L):
    if L == 1:
        ntk = np.cos(f(phi, 1)) +  (1 - phi / np.pi) * np.cos(phi)
        return ntk
    else:
        a = phi
        for i in range(L - 1):
            a = f(a, 1)
            
        ntk = np.cos(f(a, 1)) + NTK(phi, L - 1) * (1 - a / np.pi)
        return ntk
    
def get_gaussian_spectrum(ker_var, dist_var, kmax, dim):
    ## Sigma is sample variance
    ## Gamma is kernel variance
    sigma = dist_var
    gamma = ker_var
    
    a = 1/(4*sigma)
    b = 1/(2*gamma)
    c = np.sqrt(a**2 + 2*a*b)
    A = a+b+c
    B = b/A
    
    spectrum = np.array([np.sqrt(2*a/A)**(dim) * B**(k) for k in range(kmax)])
    lambda_bar = np.array([B**(k) for k in range(kmax)])
    
    degens = np.array([scipy.special.comb(k+dim-1,dim-1) for k in range(kmax)])
    
    return spectrum, degens, lambda_bar



def get_kernel_spectrum(layers, sig_w, sig_b, kmax, dim, num_pts=10000, IfNTK = True):
    alpha = dim / 2.0 - 1
    z, w = sp.special.roots_gegenbauer(num_pts, alpha)
    Q = gegenbauer.gegenbauer(z, kmax, dim)
    
    degens = np.array([gegenbauer.degeneracy_kernel(dim, k) for k in range(kmax)])
    
    kernel = np.zeros((len(layers), num_pts))
    
    L = max(layers)+1
    theta = np.arccos(z)
    KernelNTK, KernelNormalizedNTK, ThetaNTK = ntk.NTK(theta, sig_w, sig_b, L, IfNTK);
    
    for i, layer in enumerate(layers):
        kernel[i] = KernelNTK[layer]

    scaled_kernel = kernel * np.outer(np.ones(len(layers)), w)

    normalization = gegenbauer.eigenvalue_normalization(kmax, alpha, degens)

    spectrum_scaled = scaled_kernel @ Q.T / normalization
    spectrum_scaled = spectrum_scaled * np.heaviside(spectrum_scaled - 1e-20, 0)

    spectrum_true = spectrum_scaled / np.outer(len(layers), degens)

    for i in range(len(layers)):
         for j in range(kmax - 1):
            if spectrum_true[i, j + 1] < spectrum_true[i, j] * 1e-5:
                 spectrum_true[i, j + 1] = 0

    return z, spectrum_true, spectrum_scaled, degens, kernel

def exp_spectrum(s, kmax, degens):
    
    ## Here s denotes the s^(-l)

    spectrum_scaled = np.array([s**(-l) for l in range(1,kmax)])
    spectrum_scaled = np.append([1],spectrum_scaled) ## We add the zero-mode

    spectrum_true = spectrum_scaled / degens

    return spectrum_true, spectrum_scaled

def power_spectrum(s, kmax, degens):
    
    ## Here s denotes the l^(-s)

    spectrum_scaled = np.array([l**(-s) for l in range(1,kmax)])
    spectrum_scaled = np.append([1],spectrum_scaled) ## We add the zero-mode

    spectrum_true = spectrum_scaled / degens

    return spectrum_true, spectrum_scaled
    
def white_spectrum(N):
    return np.ones(N)/N

###############################################################
################# For Kernel Spectrum From Mathematica ####################
###############################################################


def ntk_spectrum(file, kmax = -1, layer = None, dim = None, return_NTK = False):
    
    ## Obtain the spectrum
    data = np.load(file, allow_pickle=True)
    eig, eig_real, eig_raw = [data['arr_'+str(i)] for i in range(len(data.files))]
    

    if(kmax != -1):
        eig = eig[:,:kmax,:]
        eig_real = eig_real[:,:kmax,:]
        eig_raw = eig_raw[:,:kmax,:]
    
    
    ## Reconstruct the NTK
    num_pts = 10000
    
    Dim = np.array([5*(i+1) for i in range(40)])
    alpha = Dim[dim] / 2.0 - 1
    z, w = sp.special.roots_gegenbauer(num_pts, alpha)
    Q = gegenbauer.gegenbauer(z, kmax, Dim[dim])
    
    k  = np.array([i for i in range(kmax)]);
    norm = (alpha+k)/alpha
    
    NTK = eig_real[dim,:,layer]*norm  @ Q
    
    if(layer != None and dim != None):
        if return_NTK:
            return eig[dim,:,layer], eig_real[dim,:,layer], NTK
        
        return eig[dim,:,layer], eig_real[dim,:,layer]
    
    if(layer != None and dim == None):
        return eig[:,:,layer], eig_real[:,:,layer]
    
    if(layer == None and dim != None):
        return eig[dim,:,:], eig_real[dim,:,:]
    
    if(layer == None and dim == None):
        return eig[:,:,:], eig_real[:,:,:]
    


def degeneracy(d,l):
    alpha = (d-2)/2
    degens = np.zeros((len(l),1))
    degens[0] = 1
    for i in range(len(l)-1):
        k = l[i+1,0]
        degens[i+1,:] = comb(k+d-3,k)*((alpha+k)/(alpha))
        
    return degens

def norm(dim,l):
    alpha = (dim-2)/2;
    area = np.sqrt(np.pi)*gamma((dim-1)/2)/gamma(dim/2);
    degen = degeneracy(dim,l)
    
    Norm = area*degen*((alpha)/(alpha+l))**2
    
    ## Also another factor of lambda/(n+lambda) comes from spherical harmoincs -> gegenbauer
    
    Norm1 = area*((alpha)/(alpha+l))*degen
    #Norm2 = area*((alpha)/(alpha+l))
    
    return [Norm1, degen]
    
def save_spectrum(directory, dim, deg, layer):

#     dim = np.array([5*(i+1) for i in range(20)])
#     deg = np.array([i+1 for i in range(100)])
#     layer = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    layer_str = [str(num) for num in layer]

    data = np.zeros((dim.size,deg.size,layer.size))
    data_real = np.zeros((dim.size,deg.size,layer.size))


    for i in range(dim.size):
        data_i = pd.read_csv(direc+str(dim[i])+".txt"
                               ,delim_whitespace=True
                               , skipinitialspace=False).T.to_numpy()
        ##Norm = np.array([norm(dim[i],d) for d in deg])
        normalization = norm(dim[i], deg.reshape(len(deg),1))
        data_real[i,:,:] = data_i / normalization[0]
        data_real[i,:,:] = data_real[i,:,:]*(data_real[i,:,:] > 1e-60)
        data[i,:,:] = data_real[i,:,:] * normalization[1]

    np.savez(directory+'GegenbauerEigenvalues.npz', data, data_real)
    
    return directory+'GegenbauerEigenvalues.npz'


###############################################################
################# Use Only These Functions ####################
###############################################################

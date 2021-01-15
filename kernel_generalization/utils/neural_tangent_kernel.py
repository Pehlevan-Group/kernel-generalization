import numpy as np

## First define the arcos kernel

def f(theta):
    f = (np.sin(theta)+(np.pi-theta)*np.cos(theta))/np.pi
    return f

def Theta(theta):
    thetaP = np.arccos(f(theta))
    return thetaP

## Given input kernel, calculate the output NNGP kernel.
## Requires the layer number.

def NNGPKernel(theta, sig_w, sig_b, L, nl_prev = None):
    kernel = 0;
    if L == 1:
        kernel = sig_w**2*np.cos(theta) + sig_b**2
    elif L > 1:
        kernel = (sig_w**2 + (L-1)*sig_b**2)*f(theta) + sig_b**2
    
    nl = sig_w**2 + L*sig_b**2;

    theta = np.arccos(kernel/nl)
    
    return kernel, theta, nl

## Given layer number, returns the kernels and theta's of all layers

def NNGP(Theta, sig_w, sig_b, L):
    
    Kernel = np.zeros((L+1,Theta.size));
    KernelNormalized = np.zeros((L+1,Theta.size));
    ThetaM = np.zeros((L+1,Theta.size));
    
    nl = sig_w**2 + sig_b**2
    Kernel[0] = sig_w**2*np.cos(Theta) + sig_b**2;
    KernelNormalized[0] = Kernel[0]/nl;
    ThetaM[0] = np.arccos(KernelNormalized[0]);
    
    kernel = Kernel[0]
    theta = ThetaM[0]
    nl = sig_w**2 + sig_b**2
    
    for i in range(1, L):
        kernel, theta, nl = NNGPKernel(theta, sig_w, sig_b, i+1, nl);
        Kernel[i] = kernel;
        KernelNormalized[i] = kernel/nl;
        ThetaM[i] = theta;
        
    return Kernel, KernelNormalized, ThetaM;

## Given input kernel, calculate the output NTK kernel.
## Requires the layer number.
## IfNTK = False makes the functions same as NNGP

def NTKernel(ntk, theta, sig_w, sig_b, l, nl_prev, IfNTK):
    # Coeff of the previous layer NNGP
    
    if l == 1:
        kernelN, thetaN, nl = NNGPKernel(theta, sig_w, sig_b, l, nl_prev);
        ntkN = kernelN
    else:
        kernelN, thetaN, nl = NNGPKernel(theta, sig_w, sig_b, l, nl_prev);
        ntkN = kernelN + IfNTK*ntk*sig_w**2*(1-theta/np.pi);
    
    if sig_w == 1:
        nl_next = nl + IfNTK*(l+l*(l+1)*sig_b**2/2-nl)
    elif sig_w > 1:
        nl_next = nl + IfNTK*(sig_w**2*(((sig_w**2))**(l)-1)/(sig_w**2-1) + l*(l+1)*sig_b**2/2-nl)
    
    #if IfNTK: assert(nl_prev + l*sig_b**2 + 1 == nl_next)
    
    return ntkN, thetaN, nl_next

def NTK(Theta, sig_w, sig_b, L, IfNTK=True):

    Kernel = np.zeros((L+1,Theta.size));
    KernelNormalized = np.zeros((L+1,Theta.size));
    ThetaM = np.zeros((L+1,Theta.size));

    nl = sig_w**2 + sig_b**2
    Kernel[0] = sig_w**2*np.cos(Theta) + sig_b**2;
    KernelNormalized[0] = Kernel[0]/nl;
    ThetaM[0] = np.arccos(KernelNormalized[0]);
    
    ntk = Kernel[0]
    theta = ThetaM[0]
    nl = sig_w**2 + sig_b**2
    
    for i in range(1,L):
        ntk,theta,nl = NTKernel(ntk, theta, sig_w, sig_b, i+1, nl, IfNTK)
        Kernel[i] = ntk;
        KernelNormalized[i] = ntk/nl;
        ThetaM[i] = theta;
        
    return Kernel, KernelNormalized, ThetaM;

def get_ntk(Theta, sig_w, sig_b, layer, IfNTK=True):
    
    L = layer +1

    Kernel = np.zeros((L+1,Theta.size));
    KernelNormalized = np.zeros((L+1,Theta.size));
    ThetaM = np.zeros((L+1,Theta.size));

    nl = sig_w**2 + sig_b**2
    Kernel[0] = np.cos(Theta) + sig_b**2;
    KernelNormalized[0] = Kernel[0]/nl;
    ThetaM[0] = np.arccos(KernelNormalized[0]);
    
    ntk = Kernel[0]
    theta = ThetaM[0]
    nl = sig_w**2 + sig_b**2
    
    for i in range(1,L):
        ntk,theta,nl = NTKernel(ntk, theta, sig_w**2, sig_b**2, i+1, nl, IfNTK)
        Kernel[i] = ntk;
        KernelNormalized[i] = ntk/nl;
        ThetaM[i] = theta;
    
    
    return Kernel[layer], KernelNormalized[layer]

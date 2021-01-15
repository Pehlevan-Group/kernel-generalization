import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
from kernel_generalization.utils import gegenbauer
import matplotlib.pyplot as plt
import numba
from numba import jit, int64

def sample_random_points(num_pts, d):
    R = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_pts)
    for i in range(num_pts):
        R[i, :] = R[i, :] / np.linalg.norm(R[i, :])
    return R

@jit(nopython=True)
def dot_prod(X,Y):
    return np.dot(X,Y)

@jit(nopython=True, parallel=True)
def compute_kernel(X, Xp, spectrum, degens, dim, kmax):

    alpha = (dim - 2) / 2
    k = np.linspace(0, kmax - 1, kmax)
    spec = spectrum * (k / alpha + 1)

    P = X.shape[0]
    Pp = Xp.shape[0]
    gram = dot_prod(X, Xp.T)
    gram = np.reshape(gram, P * Pp)
    
    Q = gegenbauer.gegenbauer(gram, kmax, dim)
    
    K = dot_prod(Q.T, spec).reshape(P, Pp)
    
    return K

def generalization(P_stu, P_teach, P_test, spectrum, degens, dim, kmax, num_repeats, lamb=0, noise_var=0):
    errors_avg = np.zeros((kmax, len(noise_var)))
    errors_tot_MC = np.zeros(len(noise_var))
    all_errs = np.zeros((num_repeats, kmax, len(noise_var)))
    all_MC = np.zeros((num_repeats, len(noise_var)))

    for i in range(num_repeats):
        #############################################
        # Define student and teacher inputs
        X_teach = sample_random_points(P_teach, dim)
        X_stu = sample_random_points(P_stu, dim)

        # Calculate the kernel Gram matrices
        K_student = compute_kernel(X_stu, X_stu, spectrum, degens, dim, kmax)
        K_stu_te = compute_kernel(X_stu, X_teach, spectrum, degens, dim, kmax)

        # Define the teacher function corrupted by noise (target function)
        sigma = np.random.normal(0, np.sqrt(noise_var*P_teach), (P_stu, len(noise_var)))
        alpha_teach = np.sign(np.random.random_sample(P_teach) - 0.5 * np.ones(P_teach))
        alpha_teach = np.outer(alpha_teach, np.ones(len(noise_var)))
        y_teach = dot_prod(K_stu_te, alpha_teach) + sigma
        

        # Calculate the regression result for student function
        K_inv = np.linalg.inv(K_student + lamb * np.eye(P_stu))
        alpha_stu = dot_prod(K_inv, y_teach)

        gram_ss = dot_prod(X_stu, X_stu.T)
        gram_st = dot_prod(X_stu, X_teach.T)
        gram_tt = dot_prod(X_teach, X_teach.T)

        Q_ss = gegenbauer.gegenbauer(gram_ss.reshape(P_stu ** 2), kmax, dim)
        Q_st = gegenbauer.gegenbauer(gram_st.reshape(P_stu * P_teach), kmax, dim)
        Q_tt = gegenbauer.gegenbauer(gram_tt.reshape(P_teach ** 2), kmax, dim)

        errors = np.zeros((kmax, len(noise_var)))
        for k in range(kmax):
            Q_ssk = Q_ss[k].reshape(P_stu, P_stu)
            Q_stk = Q_st[k].reshape(P_stu, P_teach)
            Q_ttk = Q_tt[k].reshape(P_teach, P_teach)
            a = (dim - 2) / 2
            prefactor = spectrum[k] ** 2 * (k + a) / a
            
            alpha_tt = (alpha_teach[:,0].T).dot(Q_ttk.dot(alpha_teach[:,0]))
            for n in range(len(noise_var)):
                alpha_ss = (alpha_stu[:,n].T).dot(Q_ssk.dot(alpha_stu[:,n]))
                alpha_st = (alpha_stu[:,n].T).dot(Q_stk.dot(alpha_teach[:,n]))

                errors[k,n] = prefactor * (alpha_ss - 2 * alpha_st + alpha_tt)

        errors_avg += errors / num_repeats
        all_errs[i] = errors

        X_test = sample_random_points(P_test, dim)
        K_s = compute_kernel(X_stu, X_test, spectrum, degens, dim, kmax)
        K_t = compute_kernel(X_teach, X_test, spectrum, degens, dim, kmax)

        y_s = dot_prod(K_s.T, alpha_stu)
        y_t = dot_prod(K_t.T, alpha_teach)
        tot_error = np.mean((y_s - y_t)**2, axis = 0)
        
        errors_tot_MC += tot_error/ num_repeats
        all_MC[i] = tot_error

        error_diff = np.abs(tot_error - np.sum(errors, axis = 0))/ tot_error
        curr_err = errors_tot_MC/(i+1)
            

    std_errs = np.array([sp.stats.sem(all_errs[:,:,i], axis=0) for i in range(len(noise_var))]).T
    std_MC = np.array([sp.stats.sem(all_MC[:,i]) for i in range(len(noise_var))])

    return errors_avg/P_teach, errors_tot_MC/P_teach, std_errs/P_teach, std_MC/P_teach
    
def sample_random_points_gpu(num_pts, d):
    import cupy as cp
    R = cp.random.multivariate_normal(cp.zeros(d), cp.eye(d), num_pts)
    for i in range(num_pts):
        R[i, :] = R[i, :] / cp.linalg.norm(R[i, :])
    return R

def dot_prod_gpu(X,Y):
    import cupy as cp
    return cp.dot(X,Y)

def compute_kernel_gpu(X, Xp, spectrum, degens, dim, kmax):
    import cupy as cp
    alpha = (dim - 2) / 2
    k = np.linspace(0, kmax - 1, kmax)
    spec = spectrum * (k / alpha + 1)
    P = X.shape[0]
    Pp = Xp.shape[0]
    gram = cp.dot(X, Xp.T)
    gram = cp.reshape(gram, P * Pp)
    Q = gegenbauer.gegenbauer_gpu(gram, kmax, dim)
    K = cp.dot(Q.T, cp.asarray(spec)).reshape(P, Pp)
    return K
    
def generalization_gpu(P_stu, P_teach, P_test, spectrum, degens, dim, kmax, num_repeats, lamb=0, noise_var=0):
    import cupy as cp
    errors_avg = np.zeros((kmax, len(noise_var)))
    errors_tot_MC = np.zeros(len(noise_var))
    all_errs = np.zeros((num_repeats, kmax, len(noise_var)))
    all_MC = np.zeros((num_repeats, len(noise_var)))

    for i in range(num_repeats):
        #############################################
        # Define student and teacher inputs
        X_teach = sample_random_points_gpu(P_teach, dim)
        X_stu = sample_random_points_gpu(P_stu, dim)

        # Calculate the kernel Gram matrices
        K_student = compute_kernel_gpu(X_stu, X_stu, spectrum, degens, dim, kmax)
        K_stu_te = compute_kernel_gpu(X_stu, X_teach, spectrum, degens, dim, kmax)

        # Define the teacher function corrupted by noise (target function)
        sigma = np.random.normal(0, np.sqrt(noise_var*P_teach), (P_stu, len(noise_var)))
        alpha_teach = cp.sign(cp.random.random_sample(P_teach) - 0.5 * cp.ones(P_teach))
        alpha_teach = cp.outer(alpha_teach, cp.ones(len(noise_var)))
        y_teach = cp.dot(K_stu_te, alpha_teach) + cp.asarray(sigma)
        

        # Calculate the regression result for student function
        K_inv = cp.linalg.inv(K_student + lamb * cp.eye(P_stu))
        alpha_stu = cp.dot(K_inv, y_teach)

        gram_ss = cp.dot(X_stu, X_stu.T)
        gram_st = cp.dot(X_stu, X_teach.T)
        gram_tt = cp.dot(X_teach, X_teach.T)

        Q_ss = gegenbauer.gegenbauer_gpu(gram_ss.reshape(P_stu ** 2), kmax, dim)
        Q_st = gegenbauer.gegenbauer_gpu(gram_st.reshape(P_stu * P_teach), kmax, dim)
        Q_tt = gegenbauer.gegenbauer_gpu(gram_tt.reshape(P_teach ** 2), kmax, dim)

        errors = np.zeros((kmax, len(noise_var)))
        for k in range(kmax):
            Q_ssk = Q_ss[k].reshape(P_stu, P_stu)
            Q_stk = Q_st[k].reshape(P_stu, P_teach)
            Q_ttk = Q_tt[k].reshape(P_teach, P_teach)
            a = (dim - 2) / 2
            prefactor = spectrum[k] ** 2 * (k + a) / a
            
            alpha_tt = (alpha_teach[:,0].T).dot(Q_ttk.dot(alpha_teach[:,0]))
            for n in range(len(noise_var)):
                alpha_ss = (alpha_stu[:,n].T).dot(Q_ssk.dot(alpha_stu[:,n]))
                alpha_st = (alpha_stu[:,n].T).dot(Q_stk.dot(alpha_teach[:,n]))

                errors[k,n] = prefactor * (alpha_ss - 2 * alpha_st + alpha_tt)

        errors_avg += cp.asnumpy(errors) / num_repeats
        all_errs[i] = cp.asnumpy(errors)

        X_test = sample_random_points_gpu(P_test, dim)
        K_s = compute_kernel_gpu(X_stu, X_test, spectrum, degens, dim, kmax)
        K_t = compute_kernel_gpu(X_teach, X_test, spectrum, degens, dim, kmax)

        y_s = cp.dot(K_s.T, alpha_stu)
        y_t = cp.dot(K_t.T, alpha_teach)
        tot_error = cp.mean((y_s - y_t)**2, axis = 0)
        
        errors_tot_MC += cp.asnumpy(tot_error)/ num_repeats
        all_MC[i] = cp.asnumpy(tot_error)

        error_diff = np.abs(cp.asnumpy(tot_error) - np.sum(cp.asnumpy(errors), axis = 0))/ cp.asnumpy(tot_error)
        curr_err = errors_tot_MC/(i+1)
            

    std_errs = np.array([sp.stats.sem(all_errs[:,:,i], axis=0) for i in range(len(noise_var))]).T
    std_MC = np.array([sp.stats.sem(all_MC[:,i]) for i in range(len(noise_var))])

    return errors_avg/P_teach, errors_tot_MC/P_teach, std_errs/P_teach, std_MC/P_teach
    
### On GPU
    
def sample_gaussian_points_gpu(p, dim, sigma):
    import cupy as cp
    return cp.asarray(np.random.multivariate_normal(np.zeros(dim), sigma*np.eye(dim), p))
    #return cp.random.multivariate_normal(cp.zeros(dim), sigma*cp.eye(dim), p)

def gaussian_kernel_gpu(X, Xp, ker_var):
    import cupy as cp
    G1 = cp.dot(X, X.T)
    G2 = cp.dot(Xp, Xp.T)
    G3 = cp.dot(X, Xp.T)
    R = cp.outer(cp.diag(G1), cp.ones(Xp.shape[0])) + cp.outer(cp.ones(X.shape[0]), cp.diag(G2)) - 2*G3
    K = cp.exp(-0.5*R / ker_var**2)
    
    return K

def rbf_regression_expt_gpu(pvals, pteach, dim ,gamma, sigma, lamb, num_avg, noise, errs_tot, directory = None, num_test = 1000):
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    errs = cp.zeros(len(pvals))
    stds = cp.zeros(len(pvals))
    for i in range(len(pvals)):
        p = pvals[i]
        print('p: ' + str(p), end = '\r')
        errs_i = cp.zeros(num_avg)
        for j in range(num_avg):
            X = sample_gaussian_points_gpu(p,dim,gamma)
            Xteach = sample_gaussian_points_gpu(pteach,dim,gamma)
            Xte = sample_gaussian_points_gpu(num_test,dim,gamma)
            
            alpha_t = cp.asarray(1/np.sqrt(pteach)*np.sign(np.random.random_sample(pteach) - 0.5*np.ones(pteach)))
            
            Ktarget = gaussian_kernel_gpu(X, Xteach, sigma)
            y = cp.dot(Ktarget, alpha_t) +  cp.asarray(noise*np.random.standard_normal(p))
            del Ktarget
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            K = gaussian_kernel_gpu(X, X, sigma)
            alpha = cp.dot(cp.linalg.inv(K + lamb*cp.eye(p)), y)
            del K
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            Ktest_hat = gaussian_kernel_gpu(Xte, X, sigma)
            yhat = cp.dot(Ktest_hat, alpha)
            del Ktest_hat
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            Ktest_true = gaussian_kernel_gpu(Xte, Xteach, sigma)
            y_true = cp.dot(Ktest_true, alpha_t)
            del Ktest_true
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            del X, Xteach, Xte
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            errs_i[j] = 1/num_test * cp.linalg.norm(yhat -  y_true)**2
            
        errs[i] = cp.mean(errs_i)
        stds[i] = cp.std(errs_i)
        print('p: ', p)
        if p > 1000 and directory != None:
            Errors = errs[:i]
            Std = stds[:i]/errs[:i]
            
            plt.errorbar(np.log10(pvals[:i]), np.log10(Errors), Std, fmt = '.')
            plt.plot(np.log10(pvals[:i]), np.log10(errs_tot[:i]))
            plt.show()
            
            plt.savefig(directory + 'error_curve_'+str(p)+'.pdf', bbox_inches = 'tight')

    
    return cp.asnumpy(errs), cp.asnumpy(stds)

import numpy as np
from scipy.stats import multivariate_normal, multinomial
#from model_elections import compute_qm_list
from multiprocessing import Pool
# from helper_functions import combinations
import time
from tqdm import tqdm
from EM_algorithm import EM_algorithm, get_p_est
    
def compute_q_mvn_pdf(n, p, b, parallel = False):
    M_size,G_size,I_size = b.shape[0],b.shape[1],n.shape[1]
    q = np.zeros(shape=(M_size,G_size,I_size))
    p_trunc = p[:,:-1]
    n_trunc = n[:,:-1]
    diag_p = [np.diag(p_g) for p_g in p_trunc]
    p_g_squared = np.einsum('ij,ik->ijk', p_trunc, p_trunc)
    # if M_size >= 100:
    #     parallel = True
    # if parallel:
    #     q_args = [(n[m], p, b[m], I_size, G_size, use_pdf) for m in range(M_size)]
    #     with Pool() as pool:
    #         q = pool.starmap(compute_qm_mvn, q_args)
    #     return np.array(q)
    for m in range(M_size):
        q_m = compute_qm_mvn_pdf(n_trunc[m], p, p_trunc, b[m], I_size, G_size, diag_p, p_g_squared)
        q[m] = q_m
    return q

def compute_qm_mvn_pdf(n_m_trunc, p, p_trunc, b_m, I_size, G_size, diag_p, p_g_squared):
    mu = b_m @ p_trunc   # (1,G_size) @ (G_size, I_size-1) = I_size - 1
    cov = np.diag(mu) - p_trunc.T @ np.diag(b_m) @ p_trunc # (I_size-1, I_size-1)


    covs_U = cov - diag_p + p_g_squared # (G_size, I_size-1, I_size-1)
    mus_U = mu - p_trunc # (G_size, I_size-1)
    
    vals_U, vecs_U = np.linalg.eigh(covs_U)
    
    mahas = np.zeros(shape=(G_size, I_size))
    
    inverses, invs_devs, mahas[:, I_size-1] = get_maha_mult_v2(n_m_trunc, mus_U, vals_U, vecs_U)
    
    diag_inverses = np.diagonal(inverses, axis1=1, axis2=2)


    mahas[:, :-1] = mahas[:, I_size-1][...,None] - 2 * invs_devs +  diag_inverses

    q_m = np.exp(-0.5*mahas) * p # agregar mahalanobis de T
    q_m = q_m/np.sum(q_m, axis=1)[:,None]
    

    return q_m

# https://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/ 
# https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
def get_maha_mult_v2(x, means, vals, vecs):
    # Invert the eigenvalues.
    valsinvs   = 1./vals

    devs       = x - means # G x I - 1

    valsinv_diag = [np.diag(v) for v in valsinvs]

    inverses = vecs @ valsinv_diag @ vecs.swapaxes(1,2) # G x I-1 x I-1 

    invs_devs = np.einsum('gi,gij->gj', devs, inverses) # G x I-1 (for each g you get the left hand side of mahalanobis)

    mahas = np.einsum('gi,gi->g', devs, invs_devs) # G (for each g you get the mahalanobis distance)

    return inverses, invs_devs, mahas


def compute_p(q, b):
    num = np.sum(np.multiply(q,b[...,None]),axis=0)
    dem = np.sum(b,axis=0)[...,None]
    return num / dem

def EM_mvn_pdf(X, b, convergence_value = 0.0001, max_iterations = 100, 
                p_method = 'group_proportional', load_bar = True, verbose = True,
                dict_results = {}, save_dict = False, dict_file = None):
    p_est = get_p_est(X, b, p_method)
    return EM_algorithm(X, b, p_est, compute_q_mvn_pdf, convergence_value, max_iterations, load_bar, verbose,
                        dict_results, save_dict, dict_file)
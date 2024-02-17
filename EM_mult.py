import numpy as np
import time
from tqdm import tqdm
from EM_algorithm import EM_algorithm, get_p_est

def compute_q_multinomial(n,p,b):
    p_cond = (b @ p)[:,None,:] - p[None,...] 
    q = p[None,...]*np.expand_dims(n, axis=1) / p_cond
    q[np.isnan(q)] = 0
    q = q/np.sum(q, axis=2)[:,:,None]
    q[np.isnan(q)] = 0
    return q

# def compute_q_multinomial(n,p,b):
#     p_cond = (b @ p)[:,None,:] - p[None,...] 
#     q = np.expand_dims(n, axis=1) / p_cond
#     # print('r')
#     # print(p_cond)
#     # print('q')
#     # print(q/np.sum(q, axis=2)[:,:,None])
#     # print('p')
#     # print(p[None,...])
#     # exit()
#     q[np.isnan(q)] = 0
#     q = p[None,...]*q/np.sum(q, axis=2)[:,:,None]
#     #q[np.isnan(q)] = 0
#     return q

def compute_p(q, b):
    num = np.sum(np.multiply(q,b[...,None]),axis=0)
    dem = np.sum(b,axis=0)[...,None]
    return num / dem

# (g,i) compute estimate of p using EM algorithm with parameters X and b 
def EM_mult(X, b, convergence_value = 0.0001, max_iterations = 100, 
                p_method = 'group_proportional', load_bar = True, verbose = True,
                dict_results = {}, save_dict = False, dict_file = None):
    p_est = get_p_est(X, b, p_method)
    return EM_algorithm(X, b, p_est, compute_q_multinomial, convergence_value, max_iterations, load_bar, verbose,
                        dict_results, save_dict, dict_file)



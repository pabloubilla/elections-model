import numpy as np
from helper_functions import combinations, find_tuple
from tqdm import tqdm

def compute_p_value_m(n_m, p, b_m, I_size, G_size):    
    K = [[] for f in range(G_size+1)]
    H = [[] for f in range(G_size+1)]
    T = [[] for f in range(G_size+1)]
    K[-1] = np.array(combinations(I_size,0))
    K_dict = {}
    for k_ind, k in enumerate(K[-1]):
        K_dict[-1,tuple(k)] = k_ind 
    T[-1] = np.full(len(K[-1]), 1)
    lgac_n = [sum([np.log(max(k, 1)) for k in range(n + 1)]) for n in range(np.max(b_m) + 1)]
    #hlg_p_gin = [[[n * np.log(p[g][i]) for n in range(np.max(b_m) + 1)] for i in range(I_size)] for g in range(G_size)]
    hlg_p_gin = [[[n * np.log(p[g][i]) if p[g][i] > 0 else 0.0 for n in range(np.max(b_m) + 1)] for i in range(I_size)] for g in range(G_size)]
    b_m_cum = [np.sum(b_m[:f+1]) for f in range(G_size)]
    #count = 0
    for f in range(G_size):
        K[f] = np.array(combinations(I_size,b_m_cum[f]), dtype=np.int32)
        # save the number of the tuple in the list
        # CHECK SOMETHING HAPPENED HERE
        for k_ind, k in enumerate(K[f]):
                K_dict[f,tuple(k)] = k_ind  
        H[f] = np.array(combinations(I_size,b_m[f]), dtype=np.int32)
        T[f] = np.zeros(len(K[f]))
        for k_ind in range(len(K[f])):
            k = K[f][k_ind]
            #if all(k<=n_m):
            T_k = 0.0
            for h_ind in range(len(H[f])):
                #count += 1
                h = H[f][h_ind]
                if all(h<=k):
                    #k_ind_before = find_tuple(K[f-1],k-h)
                    if f == 0:
                        k_ind_before = 0
                    else:
                        # k_ind_before_v2 = get_index(k-h,n_m,b_m_cum[f-1],I_size)
                        #k_ind_before = find_tuple(K[f-1], k-h)
                        k_ind_before = K_dict[f-1, tuple(k - h)]
                    #a = multinomial_pmf_fixed(h, n=b_m[f], p=p[f])
                    a = np.exp(lgac_n[b_m[f]] + np.sum([hlg_p_gin[f][i][h[i]] - lgac_n[h[i]] for i in range(I_size)]))
                    T_k += T[f-1][k_ind_before]*a          
                T[f][k_ind] = T_k
    p_value = 0
    # n_m_index = find_tuple(K[G_size-1], n_m) # MAYBE SHOULD REPLACE WITH K_DICT
    n_m_index = K_dict[G_size-1, tuple(n_m)] 
    p_n_m = T[G_size-1][n_m_index]
    
    total_prob = 0.0
    for k in range(len(T[G_size-1])):#itertools.chain(range(n_m_index), range(n_m_index + 1, len(T[G_size-1]))):
        p_k = T[G_size-1][k]
        total_prob += p_k
        if p_k <= p_n_m:
            p_value += p_k
    #print('G =',G_size,', Total K = ',np.sum([np.sum(len(K[f])) for f in range(G_size)]),' Total H = ',np.sum([np.sum(len(H[f])) for f in range(G_size)]), f' Basic Operation = {count}')
    #print(total_prob)               
    return max(min(p_value, 1.0), 0.0)      #, K[G_size-1]

# compute real p_value of m
def compute_p_value_m_G1(n_m, p, b_m, I_size):
    G_size = 1
    lgac_n = [sum([np.log(max(k, 1)) for k in range(n + 1)]) for n in range(np.max(b_m) + 1)]
    #hlg_p_gin = [[[n * np.log(p[g][i]) for n in range(np.max(b_m) + 1)] for i in range(I_size)] for g in range(G_size)]
    hlg_p_gin = [[[n * np.log(p[g][i]) if p[g][i] > 0 else 0.0 for n in range(np.max(b_m) + 1)] for i in range(I_size)] for g in range(G_size)]
    K = np.array(combinations(I_size,b_m[0])) 
    p_val = 0.0
    #pmf_punto = np.exp(lgac_n[b_m[0]] + np.sum([hlg_p_gin[0][i][n_m[i]] - lgac_n[n_m[i]] for i in range(I_size)]))
    pmf_punto = np.sum([hlg_p_gin[0][i][n_m[i]] - lgac_n[n_m[i]] for i in range(I_size)])
    for h in K:
        #a = np.exp(lgac_n[b_m[0]] + np.sum([hlg_p_gin[0][i][h[i]] - lgac_n[h[i]] for i in range(I_size)]))
        a = np.sum([hlg_p_gin[0][i][h[i]] - lgac_n[h[i]] for i in range(I_size)])
        if a <= pmf_punto:
            p_val += np.exp(lgac_n[b_m[0]]+a)
    return max(min(p_val, 1.0), 0.0)

# compute p_value of all ballot-boxes
def compute_p_value(n, p, b, load_bar = False):
    M_size,G_size,I_size = b.shape[0],b.shape[1],n.shape[1]
    p_value_list = []
    for m in tqdm(range(M_size), disable = not load_bar):
        #print(m)
        g_order = np.argsort(b[m]) 
        if G_size == 1: # CAMBIAR A 1
            p_value_list.append(compute_p_value_m_G1(n[m], p[g_order], b[m][g_order], I_size))
        else:
            #p_value_list.append(compute_p_value_m_comb(n[m],p[g_order],b[m][g_order],I_size,G_size) 
            p_value_list.append(compute_p_value_m(n[m],p[g_order],b[m][g_order],I_size,G_size)) 
    return np.array(p_value_list)
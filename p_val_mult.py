import numpy as np
import pickle
from tqdm import tqdm
# import binom
import scipy.stats as stats
from multiprocessing import Pool

def compute_p_value_mult(n, p, b, S = 100000, load_bar = False):
    M_size,  G_size, I_size = b.shape[0], b.shape[1], n.shape[1]
    p_value_list = []
    for m in tqdm(range(M_size), disable = not load_bar):
        r_m = (p.T @ b[m]) / np.sum(b[m])
        p_value_list.append(compute_p_value_m_mult(n[m], r_m, S))
    return np.array(p_value_list)

def compute_p_value_m_mult(n, r, S):
    J = sum(n)
    lgac_n = np.array([sum([np.log(max(k, 1)) for k in range(j + 1)]) for j in range(J+1)])
    log_p = np.log(r)
    x_samples = np.random.multinomial(J, r, size=S)
    beta_S = np.sum(x_samples * log_p, axis=1) - np.sum(lgac_n[x_samples], axis=1)
    beta_n = np.sum(n * log_p) - np.sum(lgac_n[n])
    less_p = np.sum(beta_S <= beta_n)
    return less_p/S

def compute_p_value_m_mult_threshold(x, r, S_min, S_max):
    # read threshold dict
    with open(f"thresholds/thresholds_mu6_alpha8.pickle", 'rb') as f:
        thresholds = pickle.load(f)
    J = sum(x)
    log_p = np.log(r)
    lgac_n = np.array([sum([np.log(max(k, 1)) for k in range(j + 1)]) for j in range(J+1)])
    for s in range(S_min, S_max + 1):
        n = int(10**s)
        x_samples = np.random.multinomial(J, r, size=n)
        beta_S = np.sum(x_samples * log_p, axis=1) - np.sum(lgac_n[x_samples], axis=1)
        beta_n = np.sum(x * log_p) - np.sum(lgac_n[x])
        less_p = np.sum(beta_S <= beta_n)
        if less_p >= thresholds[s]:
            break
    return less_p/n, s


def compute_pvalue_pickle(file_in, file_out, load_bar = False, S_min = 3, S_max = 8, parallel = False):
    data_out = dict()
    with open(file_in, 'rb') as f:
        data_in_pickle: dict = pickle.load(f)
    data_in = []
    for key, val in data_in_pickle.items():
        if np.any(np.isnan(val["r"])) == True:
            data_out[key] = {
            "p_value": np.nan,
            "trials": np.nan,
        }
            continue
        data_in.append((key, val))
    # data_in: list = [(key, val) for key, val in data_in.items()]

    if parallel: data_parallel = []
    for key, val in data_in:

        x = np.array(val["x"])
        r = np.array(val["r"])
        non_zero = r!=0
        non_zero = r!=0
        x = x[non_zero]
        r = r[non_zero]
        # C = len(r)
        if parallel:
            data_parallel.append((x, r, S_min, S_max))
        else:
            pval, trials = compute_p_value_m_mult_threshold(x, r, S_min, S_max)
            data_out[key] = {
                "p_value": pval,
                "trials": trials,
            }
    if parallel:
        with Pool(8) as p:
            # pval, trial
            results = p.starmap(compute_p_value_m_mult_threshold, data_parallel)

        for i, (key, val) in enumerate(data_in):
            data_out[key] = {
                "p_value": results[i][0],
                "trials": results[i][1],
            }

    # for key, val in tqdm(data_in, disable = not load_bar):
    #     if np.any(np.isnan(val["r"])) == True:
    #         data_out[key] = {
    #         "p_value": np.nan,
    #         "trials": np.nan,
    #     }
    #         continue
    #     # J = int(val["J"])
    #     x = np.array(val["x"])
    #     r = np.array(val["r"])
    #     non_zero = r!=0
    #     non_zero = r!=0
    #     x = x[non_zero]
    #     r = r[non_zero]
    #     # C = len(r)
    #     pval, trials = compute_p_value_m_mult_threshold(x, r, S_min, S_max)
    #     data_out[key] = {
    #         "p_value": pval,
    #         "trials": trials,
    #     }

    with open(file_out, 'wb') as f:
        pickle.dump(data_out, f)



def save_thresholds(S_max, mu_power, alpha_power):
    thresholds = dict()
    mu = 10**(-mu_power)
    alpha = 10**(-alpha_power)
    for i in range(1, S_max+1):
        n = int(10**(i))
        thresholds[i] = p_val_threshold_n(n, mu, alpha)
    with open(f"thresholds/thresholds_mu{mu_power}_alpha{alpha_power}.pickle", 'wb') as f:
        pickle.dump(thresholds, f)

def p_val_threshold_n(n, mu, alpha):
    cum_prob = 0
    for z in range(1, n):
        cum_prob += stats.binom.pmf(z-1, n, mu)
        if cum_prob >= 1-alpha:
            return z
        




if __name__ == '__main__':
    # pass
    # load instance with json 
    # import json
    # with open(f"instances/instance_G{3}_I{3}_M{50}_J{50}.json", 'r') as f:
    #     data = json.load(f)
    # X = np.array(data["n"])
    # b = np.array(data["b"])
    # p = np.array(data["p"])
    # print(X[0])
    # X[0,1] -= 3
    # X[0,2] += 3
    # print(compute_p_value_m_mult(X[0], p[0], 1000000))
    # print(compute_p_value_m_mult_threshold(X[0], p[0], 2, 7))

    save_thresholds(9, mu_power = 6, alpha_power = 8)


    # # print threshold pickle
    # with open(f"thresholds/thresholds_mu6_alpha8.pickle", 'rb') as f:
    #     thresholds = pickle.load(f)
    # print(thresholds)

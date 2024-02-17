from p_val import compute_p_value
from p_val_mult import compute_p_value_mult
import os
import numpy as np
import pickle

G_list = [2,3]
I_list = [2,3]
M_list = [50]
J_list = [100]
seed_list = [0]
lambda_list = [50]
method_list = ["full", "mult"]

if __name__ == '__main__':
    for j in J_list:
        for m in M_list:
            for g in G_list:
                for i in I_list:
                    for lambda_ in lambda_list:
                        result_path = f'results/J{j}_M{m}_G{g}_I{i}_lambda{lambda_}'
                        for s in seed_list:
                            p_val_dict = {}
                            for method in method_list:
                                # open pickle
                                data_path = os.path.join(result_path, f'{method}')
                                with open(f'{data_path}/{s}.pickle', 'rb') as f:
                                    data = pickle.load(f)
                                X = np.array(data["X"])
                                b = np.array(data["b"])
                                p = np.array(data["p"])
                                p_est = np.array(data["p_est"])
                                if method == 'full':
                                    p_val_dict['X'] = X
                                    p_val_dict['b'] = b
                                    p_val_dict['p'] = p
                                    p_val_dict['p_est_full'] = p_est
                                    p_val_dict['p_val_full_EM'] = compute_p_value(X, p_est, b)
                                    p_val_dict['p_val_full_real'] = compute_p_value(X, p, b)
                                    print('p_val_full_EM')
                                    print(p_val_dict['p_val_full_EM'])
                                    print('p_val_full_real')
                                    print(p_val_dict['p_val_full_real'])
                                if method == 'mult':
                                    p_val_dict['p_est_mult'] = p_est
                                    p_val_dict['p_val_mult_EM'] = compute_p_value_mult(X, p_est, b)
                                    p_val_dict['p_val_mult_real'] = compute_p_value_mult(X, p, b)
                                    print('p_val_mult_EM')
                                    print(p_val_dict['p_val_mult_EM'])
                                    print('p_val_mult_real')
                                    print(p_val_dict['p_val_mult_real'])
        
                            
                    #results\J100_M50_G2_I2_lambda0
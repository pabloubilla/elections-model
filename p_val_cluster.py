from p_val import compute_p_value
from p_val_mult import compute_p_value_mult
import os
import numpy as np
import pickle
import time

method_list = ["full", "mult"]

if __name__ == '__main__':
    J, M, G, I, lambda_, seed = os.sys.argv[1:]
    result_path = f'results/J{J}_M{M}_G{G}_I{I}_lambda{lambda_}'
    print(result_path)

    # create p_val folder if does not exist
    p_val_path = os.path.join(result_path, 'p_val')
    if not os.path.exists(p_val_path):
        os.makedirs(p_val_path)

    p_val_dict = {}
    for method in method_list:
        # open pickle
        data_path = os.path.join(result_path, f'{method}')
        with open(f'{data_path}/{seed}.pickle', 'rb') as f:
            data = pickle.load(f)
        X = np.array(data["X"])
        b = np.array(data["b"])
        p = np.array(data["p"])
        p_est = np.array(data["p_est"])
        if method == 'full':
            p_val_dict['X'] = X.copy()
            p_val_dict['b'] = b.copy()
            p_val_dict['p'] = p.copy()
            p_val_dict['p_est_full'] = p_est.copy()

            start = time.time()
            p_val_dict['p_val_full_EM'] = compute_p_value(X, p_est, b)
            p_val_dict['p_val_time_full_EM'] = time.time() - start
            print('p_val_full_EM ', p_val_dict['p_val_time_full_EM'])
            # print(p_val_dict['p_val_full_EM'])

            start = time.time()
            p_val_dict['p_val_full_real'] = compute_p_value(X, p, b)
            p_val_dict['p_val_time_full_real'] = time.time() - start
            print('p_val_full_real ', p_val_dict['p_val_time_full_real'])
            # print(p_val_dict['p_val_full_real'])

        if method == 'mult':
            p_val_dict['p_est_mult'] = p_est

            start = time.time()
            p_val_dict['p_val_mult_EM'] = compute_p_value_mult(X, p_est, b)
            p_val_dict['p_val_time_mult_EM'] = time.time() - start
            print('p_val_mult_EM ', p_val_dict['p_val_time_mult_EM'])
            # print(p_val_dict['p_val_mult_EM'])

            start = time.time()
            p_val_dict['p_val_mult_real'] = compute_p_value_mult(X, p, b)
            p_val_dict['p_val_time_mult_real'] = time.time() - start
            print('p_val_mult_real ', p_val_dict['p_val_time_mult_real'])
            # print(p_val_dict['p_val_mult_real'])
    
    # save p_val_dict into pickle
    with open(f'{p_val_path}/{seed}.pickle', 'wb') as f:
        pickle.dump(p_val_dict, f)

# p_val_full_EM 3X3  1324.9516153335571       

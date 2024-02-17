from EM_full import EM_full
from instance_maker import gen_instance_v2
import numpy as np
import json

I = 2 # candidatos (C)
G = 2 # grupos (G)
M = 50  # mesas (B)
J = 100 # personas por mesa (I)

lambda_list = [0.05*i for i in range(0,21)] # as percentage
# lambda_list = [1]
verbose = False

num_of_runs = 20
max_iterations = 1000
convergence_value = 0.001

dict_results = {}
# lambda symbol:
for lambda_ in lambda_list:
    print('lambda: ',lambda_)
    for s in range(num_of_runs):
        name_of_instance = f"J{J}_M{M}_G{G}_I{I}_lambda{int(100*lambda_)}_seed{s}"
        gen_instance_v2(G, I, M, J, lambda_ = lambda_, seed = s, name = name_of_instance, terminar = False)
        # load instance with json  
        with open(f"instances/{name_of_instance}.json", 'r') as f:
            data = json.load(f)
        X = np.array(data["n"])
        b = np.array(data["b"])
        p = np.array(data["p"])
        dict_results = {}
        dict_results['X'] = X
        dict_results['b'] = b
        dict_results['p'] = p
        dict_results['J'] = J
        dict_results['M'] = M
        dict_results['G'] = G
        dict_results['I'] = I
        dict_results['lambda_'] = lambda_
        dict_results['seed'] = s
        dict_results['method'] = 'full'

        dict_path = f'results/lambda_experiment/{name_of_instance}.pickle'

        # run EM
        EM_full(X, b, convergence_value = convergence_value, max_iterations = max_iterations, load_bar = True, verbose = verbose,
                dict_results = dict_results, save_dict = True, dict_file = dict_path)
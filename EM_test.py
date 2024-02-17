import numpy as np
import pickle
import json
import pandas as pd
import time
from instance_maker import gen_instance, gen_instance_v2
from EM_full import EM_full, compute_q_list
from EM_mvn_pdf import EM_mvn_pdf, compute_q_mvn_pdf
from EM_simulate import EM_simulate, simulate_Z
from EM_mult import EM_mult, compute_q_multinomial
from EM_mvn_cdf import EM_mvn_cdf, compute_q_mvn_cdf


if __name__ == '__main__':

    # # instance 1
    # # X = np.array([[5,5],
    # #               [6,4],
    # #               [2,8]])
    
    # # b = np.array([[4,6],
    # #               [5,5],
    # #               [6,4]])

    # # instance 2
    # X = np.array([[15,15],
    #               [16,14],
    #               [12,18]])
    
    # b = np.array([[14,16],
    #               [15,15],
    #               [16,14]])
    
    # # instance 3

    # p = np.array([[0.7,0.3],
    #               [0.3,0.7]])

    # # X = np.array([[30,10],
    # #               [10,30],
    # #               [35,5],
    # #               [5,35]])
    
    # # b = np.array([[25,15],
    # #               [14,26],
    # #               [30,10],
    # #               [9,31]])
    # print('FULL')
    # print(EM_full(X, b, verbose=False, load_bar=False))
    # print('SIMULATE')
    # print(EM_simulate(X, b, simulate=True, samples=8, verbose=False, load_bar=False))
    # print('MVN pdf')
    # print(EM_mvn_pdf(X, b, verbose=False, load_bar=False))
    # print('MVN cdf ')
    # print(EM_mvn_cdf(X, b, verbose=False, load_bar=False))
    # print('MULT')
    # print(EM_mult(X, b, verbose=False, load_bar=False))

    verbose = True
    load_bar = False
    max_iterations = 1000

    full = lambda X, b, dict_results: EM_full(X, b, dict_results = dict_results, verbose=verbose, load_bar=load_bar, max_iterations=max_iterations)
    simulate = lambda X, b, dict_results: EM_simulate(X, b, simulate = True, dict_results = dict_results, verbose=verbose, load_bar=load_bar, max_iterations=max_iterations)
    # simulate_ = lambda X, b, dict_results: EM_simulate(X, b, dict_results, simulate=True, samples=100, verbose=verbose, load_bar=load_bar, max_iterations=max_iterations)
    cdf = lambda X, b, dict_results: EM_mvn_cdf(X, b, dict_results = dict_results, verbose=verbose, load_bar=load_bar, max_iterations=max_iterations)
    pdf = lambda X, b, dict_results: EM_mvn_pdf(X, b, dict_results = dict_results, verbose=verbose, load_bar=load_bar, max_iterations=max_iterations)
    mult = lambda X, b, dict_results: EM_mult(X, b, dict_results = dict_results, verbose=verbose, load_bar=load_bar, max_iterations=max_iterations)
    # EM_methods = [full, simulate, cdf, pdf, mult]
    # EM_method_names = ["full", "simulate_100", "cdf", "pdf", "mult"]
    EM_methods = [pdf]
    EM_method_names = ["pdf"]
    # results = {}
    # df_results = pd.DataFrame(columns=["G","I","M","J","EM_method", "time","mean_error","max_error"])

    ######### EXPERIMENT 1 #########
    G = [2]
    I = [10]
    M = [50]
    J = [200]

    S = [14]

    with open(f"instances/instance_G{3}_I{3}_M{50}_J{50}.json", 'r') as f:
        data = json.load(f)
    X = np.array(data["n"])
    b = np.array(data["b"])
    p = np.array(data["p"])


    df_results = []

    for g in G:
        for i in I:
            for m in M:
                for j in J:
                    instance_array = np.zeros((len(S),len(EM_methods),4))
                    for s in S:
                        # results[g,i,m,j] = {}
                        gen_instance_v2(g, i, m, j, name=f"instance_G{g}_I{i}_M{m}_J{j}_{s}_testQ", terminar=False, seed = s)
                        # load instance with json instead of pickle
                        with open(f"instances/instance_G{g}_I{i}_M{m}_J{j}_{s}_testQ.json", 'r') as f:
                            data = json.load(f)
                        X = np.array(data["n"])
                        b = np.array(data["b"])
                        p = np.array(data["p"])
                        
                        for ix, EM_method in enumerate(EM_methods):
                            print(f"G{g}_I{i}_M{m}_J{j}_{s}_{EM_method_names[ix]}")
                            dict_results = {}
                            p_est, iters, EM_time = EM_method(X, b, dict_results=dict_results)
                            # print(dict_results['end'], dict_results['dif_Q'])
                            if dict_results['dif_Q'] < 0:
                                print(dict_results['dif_Q'])
                                print(p_est)


                        
                            # df_results = df_results._append({"G":g,"I":i,"M":m,"J":j,"EM_method":EM_method_names[ix],"p_est":p_est.tolist(),"iters":iters,"time":time,"mean_error":mean_error,"max_error":max_error}, ignore_index=True)

                            # results[g,i,m,j][EM_method_names[ix]] = {}
                            # results[g,i,m,j][EM_method_names[ix]]["p_est"] = p_est.tolist()
                            # results[g,i,m,j][EM_method_names[ix]]["iters"] = iters
                            # results[g,i,m,j][EM_method_names[ix]]["time"] = time
    # df_results = pd.DataFrame(df_results, columns=["G","I","M","J","EM_method", "mean_error","max_error","EM time", "simulation time"])
    # df_results.to_csv("results/results_exp3.csv", index=False)
    # with open(f"results/results_exp1.json", 'w') as f:
    #     json.dump(results, f)



# pre-process election data
import pickle
import pandas as pd
import time
import numpy as np
import os
from unidecode import unidecode

from model_elections import EM_algorithm, compute_q_multinomial, approx_p_G1
from poisson_multinomial import compute_q_multinomial_v2, compute_p_value_multinomial_simulate

from group_opt import new_group_matrix
from helper_functions import round_sum_n


#from multiprocessing import Pool
# # from Levenshtein import distance
# # from thefuzz import process, fuzz
from p_val_mult import compute_pvalue_pickle

verbose = True # print outputs

# function that preprocess the data
def pre_process_EM(election_name = '2021_05_CCG', parallel = False):
    # llave mesa normalizada
    llave_mesa = ['REGION', 'CIRCUNSCRIPCION SENATORIAL', 'DISTRITO', 'COMUNA', 'CIRCUNSCRIPCION ELECTORAL', 'LOCAL', 'MESA']
    llave_mesa_merge = ['CIRCUNSCRIPCION ELECTORAL', 'MESA']

    # votacion
    votes_llave_mesa = ['Region', 'Circunscripcion senatorial', 'Distrito', 'Comuna',	'Circunscripcion electoral', 'Local', 'Mesa']
    votes_columns = votes_llave_mesa + ['Nombres', 'Primer apellido', 'Segundo apellido', 'votos']
    votes_sep = '|'
    votes_path = f'{election_name}/data/{election_name}_Votacion.txt'
    votes = pd.read_csv(votes_path, sep = votes_sep, usecols = votes_columns)

    # normalize encoding of the columns content
    if verbose: print("\n######### Normalizar encoding votos #########")
    start = time.time()
    for col in votes.columns:
        if votes[col].dtype == object:
            if verbose: print("\t", col)
            votes[col] = votes[col].fillna('').apply(lambda x: unidecode(x).upper().strip())
    if verbose: print(f"Tiempo normalización: {time.time() - start} segundos")
    

    # normalize encoding of the columns names
    votes = votes.rename(columns={col: unidecode(col).upper().strip() for col in votes.columns})

    # create a column with the full name of the candidate
    votes['CANDIDATO'] = votes['NOMBRES'] + ' ' + votes['PRIMER APELLIDO'] + ' ' + votes['SEGUNDO APELLIDO']


    # votantes
    electors_llave_mesa = ['Región', 'Circunscripción senatorial', 'Distrito', 'Comuna',	'Circunscripción electoral', 'Local', 'Mesa']
    electors_columns = electors_llave_mesa + ['Rango etario', 'Sexo', 'Votantes']
    # electors_sep = ';'
    electors_path = f'{election_name}/data/{election_name}_VotantesEfectivos.xlsx'
    # read excel file
    electors = pd.read_excel(electors_path, skiprows=6)[electors_columns]

    # normalize encoding of the columns content
    print("\n### Normalizar encoding votantes ###")
    start = time.time()
    for col in electors.columns:
        if electors[col].dtype == object:
            print("\t", col)
            electors[col] = electors[col].fillna('').apply(lambda x: unidecode(x).upper().strip())
    print(f"Tiempo normalización: {time.time() - start} segundos")
    #.decode('utf-8')

    # normalize encoding of the columns names
    electors = electors.rename(columns={col: unidecode(col).upper().strip() for col in electors.columns})

    # get groups

    print("\n### Obtener grupos ###")

    start = time.time()
    electors = electors[~((electors['SEXO'] == '') & (electors['VOTANTES'] == 0))]
    group_combination = [[0,1,2,8,9,10],[3,4,5,6,7,11,12,13,14,15]] # HM <= 39 y HM >= 40
    electors['GRUPO'] = electors['RANGO ETARIO'] + ' ' + electors['SEXO'].str[0] # full name of groups
    grupos_ = np.sort(electors['GRUPO'].unique()) # order groups: first sex then age
    grupos = [g for g in grupos_ if 'H' in g] + [g for g in grupos_ if 'M' in g] 
    electors = electors.groupby(llave_mesa + ['GRUPO']).sum().reset_index() # group by GROUPS, there may be repeated rows (NACIONALIDAD)
    electors = electors.pivot(index = llave_mesa, columns = 'GRUPO', values='VOTANTES').reset_index() # long to wide
    electors = electors.reindex(llave_mesa + grupos, axis='columns') # reindex with key 
    electors[grupos] =  electors[grupos].fillna(0).astype(int) # fill na for groups who didnt vote in that ballot box
    electors_grouped, group_names_agg = new_group_matrix(electors[grupos].to_numpy(), group_combination) # get aggregated group names and aggregated matrix
    electors[group_names_agg] = electors_grouped # assign aggregated groups
    # mostrar primaras filas y todas las columnas de df electors
    print('\n### Ejemplo electores ###')
    print(electors.head(5).to_string(index=False))
    print(f'Tiempo obtener grupos: {time.time() - start} segundos')
    
    
    ## AGREGAR GRUPOS ##


    # definir variables para niveles de agregacion


    ### RUN EM ###

    print("\n### Ejecutar EM ###")

    nivel_agregacion_candidatos = 'DISTRITO'
    niveles_agregacion_candidatos = votes[nivel_agregacion_candidatos].unique() # ¿ MAS DE UNA DIMENSION?

    nivel_agregacion_EM = ['CIRCUNSCRIPCION ELECTORAL', 'LOCAL']

    wrong_locs = [] # guardar casos con error

    p_values = []


    dict_dfs_distritos = {}

    dict_input_pvalue = {}

    # # EM algorithm
    # def run_EM():

    G = len(group_names_agg)
    
    # measure times
    sequential_time = 0
    parallel_time = 0

    for d in niveles_agregacion_candidatos:

        sub_votes = votes[votes['DISTRITO'] == d].copy()

        sub_electors = electors[electors['DISTRITO'] == d].copy()

        candidatos = sub_votes['CANDIDATO'].unique()

        sub_votes = sub_votes.pivot(index = llave_mesa, columns='CANDIDATO', values='VOTOS').reset_index()
        
        niveles_agregacion_EM = sub_votes[nivel_agregacion_EM].drop_duplicates().values.tolist()

        # merge
        df_distrito = sub_votes.merge(sub_electors[llave_mesa_merge + group_names_agg], on = llave_mesa_merge, how = 'inner')    
        
        I = len(candidatos)

        # empty probs
        for g in range(G):
            for i in range(I):
                df_distrito.loc[:,f'P_{candidatos[i]}_{group_names_agg[g]}'] = 0
        
        for i in range(I):
            df_distrito.loc[:,f'E_{candidatos[i]}'] = -1

        df_distrito['NUM MESAS'] = -1
        ### non parallel ##

        start = time.time()
        for l in niveles_agregacion_EM:
            index_local = (df_distrito['CIRCUNSCRIPCION ELECTORAL'] == l[0]) & (df_distrito['LOCAL'] == l[1])
            df_local = df_distrito[index_local].copy()
            
            x = df_local[candidatos].to_numpy()
            b = df_local[group_names_agg].to_numpy()
            
            Js = np.sum(x, axis = 1)

            M = x.shape[0]

            p, iterations, t = EM_algorithm(x, b, compute_q_multinomial_v2, simulate=False, p_method='group_proportional', max_iterations=200, verbose = False, load_bar = False)
            
            for g in range(G):
                for i in range(I):
                    df_distrito.loc[index_local,f'P_{candidatos[i]}_{group_names_agg[g]}'] = p[g,i]

            #print(p)
            #print(b)
            # REVISAR, b NO SIEMPRE ES IGUAL A x
            r = b @ p / np.sum(b, axis = 1)[...,None] #m,i
            E_l = r * Js[...,None]
            if ~np.any(np.isnan(p)):
                # E_l = np.zeros((M, I))
                # for i in range(I):
                #     #print(b)
                #     E_l[:,i] = b @ p[:,i] # mg x g
                #     #print(E_l)
                for m in range(M):

                    E_l[m] = round_sum_n(E_l[m])

                df_distrito.loc[index_local,[f'E_{candidatos[i]}' for i in range(I)]] = E_l
                
            df_distrito.loc[index_local, 'NUM MESAS'] = df_local.shape[0]

            df_distrito.loc[index_local, 'NUM VOTOS'] = Js

            # if ~np.isnan(p[0,0]):
            for m in range(M):

                r_m = (p.T @ b[m]) / np.sum(b[m])
                mesa = df_local.iloc[m]['MESA']
                dict_input_pvalue[l[0], mesa] = { 'r': r_m,
                                        'J': np.sum(x[m]),
                                        'x': x[m]}
                    # p_value_m = compute_p_value_multinomial_simulate(x[m], p, b[m], 100)
                    # p_values.append(p_value_m)
        sequential_time += time.time() - start

        # parallel

        # if parallel:

        #     start = time.time()

        #     x = []
        #     b = []
        #     indexes_local = []
        #     for l in niveles_agregacion_EM:
        #         index_local = (df_distrito['CIRCUNSCRIPCION ELECTORAL'] == l[0]) & (df_distrito['LOCAL'] == l[1])
        #         indexes_local.append(index_local)
        #         df_local = df_distrito[index_local].copy()
        #         # electors_l = sub_electors[(sub_electors['CIRCUNSCRIPCION ELECTORAL'] == l[0]) & (sub_electors['LOCAL'] == l[1])].copy()
        #         # votes_l = sub_votes[(sub_votes['CIRCUNSCRIPCION ELECTORAL'] == l[0]) & (sub_votes['LOCAL'] == l[1])].copy()

        #         x.append(df_local[candidatos].to_numpy())
        #         b.append(df_local[group_names_agg].to_numpy())

        
        #     # (X, b, q_method = compute_q_list, convergence_value = 0.0001, max_iterations = 100, 
        #     # p_method = 'proportional', simulate = False, Z = None, load_bar = True, print_p = True)
            
        #     EM_args = [(x[a], b[a], compute_q_multinomial_v2, 0.0001, 200, 'group_proportional', False, None, False, False) for a in range(len(x))]
            
        #     with Pool() as pool:
        #         pool.starmap(EM_algorithm, EM_args)
            
        #     parallel_time += time.time() - start


        # save file for distrito
        path_distrito = f'{election_name}/output/{d}'
        if not os.path.exists(path_distrito):
            os.makedirs(path_distrito)
        df_distrito.to_csv(os.path.join(path_distrito,f'{d}.csv'), index=False)
        dict_dfs_distritos[d] = df_distrito

    # print times for EM
    print(f'Sequential time: {sequential_time}')
    if parallel:
        print(f'Parallel time: {parallel_time}')

    file_dict_dfs_distritos = f'{election_name}/output/{election_name}_dfs_distritos.pickle'
    with open(file_dict_dfs_distritos, 'wb') as handle:
        pickle.dump(dict_dfs_distritos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    file_dict_input_pvalues = f'{election_name}/output/{election_name}_input_pvalues.pickle'
    with open(file_dict_input_pvalues, 'wb') as handle:
        pickle.dump(dict_input_pvalue, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


            # try:  
            #     # G = 1 
            #     # p = approx_p_G1(X) 

            #     # G > 1
            #     p, iterations, time = EM_algorithm(X, b, compute_q_multinomial, simulate=False, p_method='group_proportional', max_iterations=200, print_p = False)

            # except:
            #     wrong_locs.append(l)
            
#run_EM()


# SAVE PICKLE FILE
# import pickle
# with open('dict.pickle', 'wb') as handle:
#     pickle.dump(EM_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # READ PICKLE FILE
# with open('2021_05_CCG_EM.pickle', 'rb') as handle:
#     EM_dict = pickle.load(handle)

def run_pvalue(election_name):
    file_dict_input_pvalues = f'{election_name}/output/{election_name}_input_pvalues.pickle'
    file_dict_output_pvalues = f'{election_name}/output/{election_name}_output_pvalues.pickle'
    compute_pvalue_pickle(file_in = file_dict_input_pvalues, file_out = file_dict_output_pvalues)


def add_pvalues(election_name):
    
    # read output pvalues
    file_dict_output_pvalues = f'{election_name}/output/{election_name}_output_pvalues.pickle'
    with open(file_dict_output_pvalues, 'rb') as handle:
        p_value_pickle = pickle.load(handle)

    # read dataframe of districts from EM
    file_dict_dfs_distritos = f'{election_name}/output/{election_name}_dfs_distritos.pickle'
    with open(file_dict_dfs_distritos, 'rb') as handle:
        dict_dfs_distritos = pickle.load(handle)

    if verbose: print("#### Creando .csv para cada distrito y país. ####")
    llave_mesa = ['REGION', 'CIRCUNSCRIPCION SENATORIAL', 'DISTRITO', 'COMUNA', 'CIRCUNSCRIPCION ELECTORAL', 'LOCAL', 'MESA']
    df_pais_list = []

    lista_distritos = sorted(list(dict_dfs_distritos.keys()))
    with open(f'{election_name}/output/DISTRITOS.txt', "w") as output:
        output.write('\n'.join(map(str, lista_distritos)))

    for d in lista_distritos:
        # d = 'DISTRITO 2'
        # df_distrito = dict_dfs_distritos[d]
        df_distrito = dict_dfs_distritos[d].copy()
        df_distrito['P-VALOR'] = 0.0
        for index_m, df_distrito_row in df_distrito.iterrows():
            m = df_distrito_row['MESA']
            c = df_distrito_row['CIRCUNSCRIPCION ELECTORAL']
            df_distrito.loc[index_m, 'P-VALOR'] = p_value_pickle[(c,m)]['p_value']
            print(p_value_pickle[(c,m)]['p_value'])
        df_distrito.to_csv(f'{election_name}/output/{d}/{d}.csv', index=False)
        df_pais_list.append(df_distrito[llave_mesa+['NUM MESAS', 'P-VALOR']])


        # csv por local
        loc_list = list(df_distrito['LOCAL'].unique())

        for l in loc_list:
            df_local = df_distrito[df_distrito['LOCAL'] == l].copy()
            # NOT CATCHING ANY (????)
            if np.any(df_local['P-VALOR'] <= 1e-5):
                df_local.to_csv(f'{election_name}/output/{d}/{l}.csv')
            else:
                loc_list.remove(l)
        
        # guardar locales con mesas extrañas
        with open(f'{election_name}/output/{d}/LOCALES.txt', "w") as output:
            output.write('\n'.join(map(str, loc_list)))



    df_pais = pd.concat(df_pais_list)
    df_pais.to_csv(f'{election_name}/output/{election_name}_PAIS.csv', index=False)
    
    return 0 
    #[('OSORNO', '99M')]
if __name__ == '__main__':
    start_total = time.time()
    election_name = '2021_05_CCG'
    # dict_dfs_distritos = 
    #pre_process_EM(election_name=election_name)
    #run_pvalue(election_name)
    add_pvalues(election_name)
    print(f'Total time: {time.time() - start_total}')

# import numpy as np
# stop = False
# for key in EM_dict:
#     if np.min(EM_dict[key]['r']) == 0:
#         zero_index = np.where(EM_dict[key]['r'] == 0)[0]
#         print(zero_index)
#         for i in zero_index:
#             stop = True
#             if EM_dict[key]['x'][i] != 0:
#                 print(key)
#                 print(EM_dict[key]['x'])
#                 print(EM_dict[key]['r'])
#                 stop = True
#         # if np.min(EM_dict[key]['x']) != 0:
#         #     print(key)
#         #     print(EM_dict[key]['x'])
#         #     print(EM_dict[key]['r'])
#         #     stop = True
#     if stop:
#         break
            


# electors[electors['CIRCUNSCRIPCION ELECTORAL'] == 'PEMUCO']['MESA'].unique()
# votes[votes['CIRCUNSCRIPCION ELECTORAL'] == 'PEMUCO']['MESA'].unique()
# HAY VOTOS PERO NO HAY VOTANTES: MESA 14V-15 DE PEMUCO
# REGION                                                            DE NUBLE
# CIRCUNSCRIPCION SENATORIAL                   CIRCUNSCRIPCION SENATORIAL 16
# DISTRITO                                                       DISTRITO 19
# COMUNA                                                              PEMUCO
# CIRCUNSCRIPCION ELECTORAL                                           PEMUCO
# LOCAL                         LICEO POLIVALENTE TOMAS ARNALDO HERRERA VEGA
# MESA                                                                14V-15


# /opt/homebrew/bin/python3.9
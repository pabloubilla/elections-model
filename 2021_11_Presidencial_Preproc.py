# pre-process election data
import pickle
import pandas as pd
import time
import numpy as np
import os   
from unidecode import unidecode
from natsort import natsorted # ordenar por número
import re

import sys
from p_val_mult import compute_pvalue_pickle
# from pvalues_voting_main.p_val_mult_C import compute_pvalue_pickle_C # C function not implemented yet

from EM_mult import EM_mult
from group_opt import new_group_matrix, optimze_age_groups_v2, group_election_routine
from helper_functions import round_sum_n

verbose = True # print outputs

# function that preprocess the data
def pre_process_EM(election_name):
    # function that correct names
    def correct_names(col_name):
        if col_name.lower() == "dist":
            return "DISTRITO"
        elif col_name.lower() == "dist_cod":
            return "DISTRITO CODIGO"
        elif col_name.lower() == "circ":
            return "CIRCUNSCRIPCION ELECTORAL"
        elif col_name.lower() == "circ_cod":
            return "CIRCUNSCRIPCION ELECTORAL CODIGO"
        elif col_name.lower() == "local_cod":
            return "LOCAL CODIGO"
        else:
            return unidecode(col_name).upper().strip()

    # election_name = "2023_05_CCG"
    # CIRCUNSCRIPCION SENATORIAL : son 16 (se usa para eleccion de sandores)
    # DISTRITO : son 28 (se usan para eleccion de diputados)
    # CIRCUNSCRIPCION ELECTORAL : son 346 en la eleccion de convencioanles del 2023

    # llave mesa normalizada
    llave_mesa = ['REGION', 'CIRCUNSCRIPCION ELECTORAL', 'LOCAL', 'MESA']
    # columnas_complementarios = ['DESCUADRADA', 'ERROR']
    columnas_complementarios = []
    # llave_mesa_merge = ['CIRCUNSCRIPCION', 'MESA']

    # for file in os.listdir(f'{election_name}/mesas_gen/'):
    #     print("file = ", file)
    # exit(1)
    # votes = pd.concat([pd.read_csv(f'{election_name}/mesas_gen/{file}', sep = ';', usecols = ['Lista',	'Partido',	'Nombre',	'Votos',	'dist',	'dist_cod',	'circ',	'circ_cod',	'local',	'local_cod',	'mesa',	'descuadrada', 'error']) for file in os.listdir(f'{election_name}/mesas_gen/') if file.endswith('.csv')])
    if verbose: print("\n######### Leer votos #########")
    votes = pd.read_excel(f'{election_name}/data/{election_name}.xlsx', sheet_name='Votación en Chile', skiprows = 6)
    # if verbose: print(votes.head())

    # normalize encoding of the columns content
    if verbose: print("\n######### Normalizar encoding votos #########")
    start = time.time()
    for col in votes.columns:
        if votes[col].dtype == object:
            if verbose: print("\t", col)
            votes[col] = votes[col].fillna('').apply(lambda x: re.sub(' +', ' ', unidecode(x).upper().strip().replace('"','')))
    if verbose: print(f"Tiempo normalización: {time.time() - start} segundos")
    votes = votes.rename(columns={col: unidecode(col).upper().strip() for col in votes.columns})
    

    # create a column with the full name of the candidate
    votes['CANDIDATO'] = votes['NOMBRES'] + ' ' + votes['PRIMER APELLIDO']
    # only add CANDIDATO if PRIMER APELLIDO is not empty
    votes.loc[votes['PRIMER APELLIDO'] == '', 'CANDIDATO'] = votes['NOMBRES']
    candidatos = natsorted(votes['CANDIDATO'].unique())
    # PIVOT
    votes = votes.pivot(index=llave_mesa, columns='CANDIDATO', values='VOTOS').reset_index()
    votes = votes.rename(columns={col: unidecode(col).upper().strip() for col in votes.columns})
    votes['NULO BLANCO'] = votes['VOTOS NULOS'] + votes['VOTOS EN BLANCO'] # unir votos nulos y votos en blanco
    candidatos = candidatos + ['NULO BLANCO']
    candidatos = [c for c in candidatos if c not in ['VOTOS NULOS', 'VOTOS EN BLANCO']] # no considerar votos nulos ni en blanco, solo la suma

    # save file with votes
    votes.to_csv(f'{election_name}/output/{election_name}_VOTOS.csv', index=False)
    
    if verbose: print(votes.head())

    # guardar candidatos
    lista_candidatos = natsorted(list(candidatos))
    with open(f'{election_name}/output/CANDIDATOS.pickle', "wb") as handle:
        pickle.dump(lista_candidatos, handle)


    # votantes
    # read excel file
    if verbose: print("\n######### Leer votantes #########")
    electors = pd.read_excel(f'{election_name}/data/{election_name}.xlsx', skiprows=6, sheet_name = 'Votantes efectivos en Chile')

    # normalize encoding of the columns content
    if verbose: print("\n### Normalizar encoding votantes ###")
    start = time.time()
    for col in electors.columns:
        if electors[col].dtype == object:
            print("\t", col)
            electors[col] = electors[col].fillna('').apply(lambda x: re.sub(' +', ' ', unidecode(x).upper().strip().replace('"','')))
    if verbose: print(f"Tiempo normalización: {time.time() - start} segundos")
    #.decode('utf-8')

    # normalize encoding of the columns names
    electors = electors.rename(columns={col: unidecode(col).upper().strip() for col in electors.columns})

    # get groups

    if verbose: print("\n### Obtener grupos ###")

    start = time.time()
    electors = electors[~((electors['RANGO ETARIO'] == '') & (electors['VOTANTES'] == 0))]
    # group_combination = [[0,1,2,8,9,10],[3,4,5,6,7,11,12,13,14,15]] # HM <= 39 y HM >= 40
    # group_combination = [[0,1,8,9],[2,3,10,11],[4,5,6,7,12,13,14,15]] # <= 29, <= 49, 50+
    # group_combination = [[i, i+8] for i in range(8)] # all ages
    # G = len(group_combination) # number of groups

    # WE ARE CONSIDERING ONLY AGE GROUPS NOW
    electors['GRUPO'] = electors['RANGO ETARIO'] # + ' ' + electors['SEXO'].str[0] # full name of groups
    grupos = list(np.sort(electors['GRUPO'].unique())) # order groups: first sex then age
    # grupos = [g for g in grupos_ if 'H' in g] + [g for g in grupos_ if 'M' in g] 
    electors = electors.groupby(llave_mesa + ['GRUPO']).sum().reset_index() # group by GROUPS, there may be repeated rows (NACIONALIDAD)
    electors = electors.pivot(index = llave_mesa, columns = 'GRUPO', values='VOTANTES').reset_index() # long to wide
    electors = electors.reindex(llave_mesa + grupos, axis='columns') # reindex with key 
    electors[grupos] =  electors[grupos].fillna(0).astype(int) # fill na for groups who didnt vote in that ballot box
    
    # IF YOU WANT TO GROUP BEFORE
    # electors_grouped, group_names_agg = new_group_matrix(electors[grupos].to_numpy(), group_combination) # get aggregated group names and aggregated matrix
    # electors[group_names_agg] = electors_grouped # assign aggregated groups
    # mostrar primaras filas y todas las columnas de df electors

    # save file with electors
    electors.to_csv(f'{election_name}/output/{election_name}_ELECTORES.csv', index=False)
    # guardar grupos
    lista_grupos = list(natsorted(grupos))
    with open(f'{election_name}/output/GRUPOS.pickle', "wb") as handle:
        pickle.dump(lista_grupos, handle)

    if verbose: print('\n### Ejemplo electores ###')
    print(electors.head(5).to_string(index=False))
    if verbose: print(f'Tiempo obtener grupos: {time.time() - start} segundos')


def run_EM(election_name):
    # llave mesa normalizada
    llave_mesa = ['REGION', 'CIRCUNSCRIPCION ELECTORAL', 'LOCAL', 'MESA']
    
    votes = pd.read_csv(f'{election_name}/output/{election_name}_VOTOS.csv')
    electors = pd.read_csv(f'{election_name}/output/{election_name}_ELECTORES.csv')
    
    # read candidatos
    with open(f'{election_name}/output/CANDIDATOS.pickle', "rb") as handle:
        candidatos = pickle.load(handle)
    # read grupos
    with open(f'{election_name}/output/GRUPOS.pickle', "rb") as handle:
        grupos = pickle.load(handle)

    if verbose: print("\n######### Ejecutar EM #########")

    # unidad maxima de los mismos candidatos
    nivel_agregacion_candidatos = 'REGION'
    niveles_agregacion_candidatos = votes[nivel_agregacion_candidatos].unique() # ¿ MAS DE UNA DIMENSION?

    # con que mesas en conjunto se estima el "p"
    nivel_agregacion_EM = ['CIRCUNSCRIPCION ELECTORAL']

    wrong_locs = [] # guardar casos con error

    dict_dfs_distritos = {}

    dict_input_pvalue = {}

    sequential_time = 0


    # group_combination = [[0,1,2,8,9,10],[3,4,5,6,7,11,12,13,14,15]] # HM <= 39 y HM >= 40
    # group_combination = [[0,1,8,9],[2,3,10,11],[4,5,6,7,12,13,14,15]] # <= 29, <= 49, 50+
    # electors_grouped, group_names_agg = new_group_matrix(electors[grupos].to_numpy(), group_combination) # get aggregated group names and aggregated matrix
    # electors[group_names_agg] = electors_grouped # assign aggregated groups
    

    failed_circs = []

    for d in niveles_agregacion_candidatos:

        # save file for distrito
        path_distrito = f'{election_name}/output/{d}'
        if not os.path.exists(path_distrito):
            os.makedirs(path_distrito)
        


        sub_votes = votes[votes[nivel_agregacion_candidatos] == d].copy()

        # pasar a tabla horizontal con candidatos en columnas, cada fila es una mesa
        #  + ['ERROR', 'DESCUADRADA']
        # sub_votes = sub_votes.pivot(index = llave_mesa + columnas_complementarios, columns='CANDIDATO', values='VOTOS').reset_index()
        # sub_votes = sub_votes[llave_mesa + columnas_complementarios + candidatos]
        #sub_votes = sub_votes.pivot(index = llave_mesa, columns='CANDIDATO', values='VOTOS').reset_index()
    
        # sub_votes.to_csv("abc.csv")



        niveles_agregacion_EM = sub_votes[nivel_agregacion_EM].drop_duplicates().values.tolist()

        # if G == 1:
        #     df_distrito = sub_votes.copy()
        # else:
            # merge
        sub_electors = electors[electors['REGION'] == d].copy()
        df_distrito = sub_votes.merge(sub_electors[llave_mesa + grupos], on = llave_mesa, how = 'inner')    
        
        I = len(candidatos)


        # numero de votantes
        df_distrito['NUM VOTOS'] = df_distrito[candidatos].sum(axis=1)

        # eliminamos mesas sin votos
        df_distrito = df_distrito[df_distrito['NUM VOTOS'] > 0]

        df_distrito['NUM MESAS'] = -1

        df_distrito['P-VALOR'] = -1
        df_distrito['LOG P-VALOR'] = -1

        # # empty probs
        # for g in range(G):
        #     for i in range(I):
        #         df_distrito.loc[:,f'P_{candidatos[i]}_{group_names_agg[g]}'] = 0
        
        for i in range(I):
            df_distrito.loc[:,f'E_{candidatos[i]}'] = -1


        
        start = time.time()
        for l in niveles_agregacion_EM:
            path_circ = f'{election_name}/output/{d}/{l[0]}'
            if not os.path.exists(path_circ):
                os.makedirs(path_circ)
        

            if verbose: print(f'\t{l[0]}')
            index_local = (df_distrito['CIRCUNSCRIPCION ELECTORAL'] == l[0])
            df_local = df_distrito.loc[index_local].copy()
            if len(df_local) == 0: # no hay mesas en alguna de las bases
                continue

            x = df_local[candidatos].to_numpy()
            
            Js = df_local['NUM VOTOS'].to_numpy()

            # b = np.array(Js, dtype = int)[...,None] if G == 1 else df_local[group_names_agg].to_numpy()
            b_ = df_local[grupos].to_numpy()

            M = x.shape[0]

            # if l[0] == 'CODPA':
            #     print(df_local)
            #     print(x)
            #     print('')
            #     print(b_)
            #     exit()
            # if G == 1:
            #     p = approx_p_G1(x)[None,...]
            # else:
            try:
                # group_combinations = optimze_age_groups_v2(b_, min_std = 0.1, max_corr = 1, add_std = True)
                # group_combinations, _ = group_election_routine(x, b_, std_thres=0.05, threshold='g')
                if M > 1:
                    group_combinations, _ = group_election_routine(x, b_, std_thres=0.05, threshold='gc') 
                else: 
                    group_combinations = [[0,1,2,3,4,5,6,7]]
            except:
                group_combinations = [[0,1,2,3,4,5,6,7]]
                failed_circs.append((l[0], M))
            b, group_names_agg = new_group_matrix(b_, group_combinations) # get aggregated group names and aggregated matrix

            df_local[group_names_agg] = b # assign aggregated groups

            G = len(group_names_agg) # number of groups
            
            p, iterations, t = EM_mult(x, b, verbose = False, max_iterations = 1000)
            
            for g in range(G):
                for i in range(I):
                    df_local[f'P_{candidatos[i]}_{group_names_agg[g]}'] = p[g,i]

            df_p_local = pd.DataFrame(p, columns = candidatos, index = group_names_agg)
            df_p_local.to_csv(os.path.join(path_circ,f'P_{l[0]}.csv'))
            

            #print(p)
            #print(b)
            # REVISAR, b NO SIEMPRE ES IGUAL A x
            r = b @ p / np.sum(b, axis = 1)[...,None] #m,i
            E_l = r * Js[...,None]
   

            if ~np.any(np.isnan(p)):
                for m in range(M):
                    E_l[m] = round_sum_n(E_l[m])
                df_local.loc[:,[f'E_{candidatos[i]}' for i in range(I)]] = E_l
                df_local.loc[:,[f'DIF_{candidatos[i]}' for i in range(I)]] = x - E_l
                
            df_local['NUM MESAS'] = M
            df_distrito.loc[index_local, 'NUM MESAS'] = M

            df_local.to_csv(os.path.join(path_circ,f'{l[0]}.csv'), index=False)

            # save GRUPOS 
            grupos_local = list(natsorted(group_names_agg))
            with open(os.path.join(path_circ, 'GRUPOS.pickle'), "wb") as handle:
                pickle.dump(grupos_local, handle)

            # df_distrito.loc[index_local, 'NUM VOTOS'] = Js

            # if ~np.isnan(p[0,0]):
            for m in range(M):

                r_m = (p.T @ b[m]) / np.sum(b[m])
                mesa = df_local.iloc[m]['MESA']
                dict_input_pvalue[l[0], mesa] = { 'r': r_m,
                                        'J': np.sum(x[m]),
                                        'x': x[m]}
                    # p_value_m = compute_p_value_multinomial_simulate(x[m], p, b[m], 100)
                    # p_values.append(p_value_m)




        df_distrito.to_csv(os.path.join(path_distrito,f'{d}.csv'), index=False)
        dict_dfs_distritos[d] = df_distrito

    print(f'Failed circs: {failed_circs}')
    sequential_time = time.time() - sequential_time

    # print times for EM
    print(f'Sequential time: {sequential_time}')
    
    file_dict_dfs_distritos = f'{election_name}/output/{election_name}_dfs_distritos.pickle'
    with open(file_dict_dfs_distritos, 'wb') as handle:
        pickle.dump(dict_dfs_distritos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    file_dict_input_pvalues = f'{election_name}/output/{election_name}_input_pvalues.pickle'
    with open(file_dict_input_pvalues, 'wb') as handle:
        pickle.dump(dict_input_pvalue, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0

def run_pvalue(election_name, pval_parallel, use_C):
    print("\n######### Compute p-values #########")
    c_string = '_C'
    file_dict_input_pvalues = f'{election_name}/output/{election_name}_input_pvalues.pickle'
    file_dict_output_pvalues = f'{election_name}/output/{election_name}_output_pvalues{c_string*use_C}.pickle'
    if use_C:
        # compute_pvalue_pickle_C(file_in = file_dict_input_pvalues, file_out = file_dict_output_pvalues)
        # compute_pvalue_pickle_C() # gotta add the C function
        pass
    else:
        compute_pvalue_pickle(file_in = file_dict_input_pvalues, file_out = file_dict_output_pvalues, load_bar=True , S_min = 2, S_max = 8, parallel=pval_parallel)


def add_pvalues(election_name):
    print("\n######### creating output files #########")
    
    # read output pvalues
    file_dict_output_pvalues = f'{election_name}/output/{election_name}_output_pvalues_C.pickle'
    with open(file_dict_output_pvalues, 'rb') as handle:
        p_value_pickle = pickle.load(handle)

    # read dataframe of districts from EM
    file_dict_dfs_distritos = f'{election_name}/output/{election_name}_dfs_distritos.pickle'
    with open(file_dict_dfs_distritos, 'rb') as handle:
        dict_dfs_distritos = pickle.load(handle)

    if verbose: print("#### Creando .csv para cada distrito y país. ####")
    # llave_mesa = ['REGION', 'CIRCUNSCRIPCION SENATORIAL', 'DISTRITO', 'COMUNA', 'CIRCUNSCRIPCION ELECTORAL', 'LOCAL', 'MESA']
    llave_mesa = ['REGION', 'CIRCUNSCRIPCION ELECTORAL', 'LOCAL', 'MESA']
    # columnas_complementarios = ['DESCUADRADA', 'ERROR']
    columnas_complementarios = []
    df_pais_list = []

    lista_distritos = sorted(list(dict_dfs_distritos.keys()))
    with open(f'{election_name}/output/REGIONES.txt', "w") as output:
        output.write('\n'.join(map(str, lista_distritos)))

    for d in lista_distritos:
        # d = 'DISTRITO 2'
        # df_distrito = dict_dfs_distritos[d]
        df_distrito = dict_dfs_distritos[d].copy()
        # df_distrito['P-VALOR'] = 0.0
        for index_m, df_distrito_row in df_distrito.iterrows():
            m = df_distrito_row['MESA']
            c = df_distrito_row['CIRCUNSCRIPCION ELECTORAL']
            df_distrito.loc[index_m, 'P-VALOR'] = p_value_pickle[(c,m)]['p_value']
            # print(p_value_pickle[(c,m)]['p_value'])
        df_distrito['LOG P-VALOR'] = np.log10(df_distrito['P-VALOR'])
        df_distrito.to_csv(f'{election_name}/output/{d}/{d}.csv', index=False)
        df_pais_list.append(df_distrito[llave_mesa + ['NUM MESAS', 'NUM VOTOS'] + columnas_complementarios + ['P-VALOR', 'LOG P-VALOR']])
        # df_pais_list.append(df_distrito[llave_mesa + ['NUM MESAS', 'NUM VOTOS', 'P-VALOR', 'LOG P-VALOR']])
        # P-VALOR Ordenado

        # csv por local
        # loc_list = list(df_distrito['LOCAL'].unique())
        circ_list = list(df_distrito['CIRCUNSCRIPCION ELECTORAL'].unique())

        # guardar circunscripciones
        with open(f'{election_name}/output/{d}/CIRCUNSCRIPCIONES.txt', "w") as output:
            output.write('\n'.join(map(str, circ_list)))


        for c in circ_list:
            break 
            # don't make circ path

            # path_circ = f'{election_name}/output/{d}/{c}'
            # if not os.path.exists(path_circ):
            #     os.makedirs(path_circ)
            df_circ = df_distrito[df_distrito['CIRCUNSCRIPCION ELECTORAL'] == c]
            df_circ.to_csv(f'{election_name}/output/{d}/{c}.csv')
            
            loc_list = list(df_circ['LOCAL'].unique())

            for l in loc_list:
                # no guardar local
                pass
                # df_local = df_circ[df_circ['LOCAL'] == l].copy()
                # NOT CATCHING ANY (????)
                # if np.any(df_local['P-VALOR'] <= 1e-5):
                # df_local.to_csv(f'{election_name}/output/{d}/{c}/{l}.csv')
                # else:
                #     loc_list.remove(l)
        
             # guardar locales con mesas extrañas
            # with open(f'{election_name}/output/{d}/{c}/LOCALES.txt', "w") as output:
            #     output.write('\n'.join(map(str, loc_list)))



    df_pais = pd.concat(df_pais_list)
    # df_pais['ORDER P-VALOR'] = compute_order_statistics_normal(df_pais['P-VALOR'].to_list())
    df_pais.to_csv(f'{election_name}/output/{election_name}_PAIS.csv', index=False)
    
    return None


if __name__ == '__main__':
    start_total = time.time()
    election_name = "2021_11_Presidencial"
    use_C = True
    pval_parallel = True
    # dict_election = election_parameters(election_name)
    
    # if verbose: print('PROCESS EM')
    # pre_process_EM(election_name)

    # if verbose: print('RUN EM')
    # run_EM(election_name)

    # if verbose: print('RUN P-VAL')
    # run_pvalue(election_name, pval_parallel, use_C)

    if verbose: print('ADD P-VAL')
    add_pvalues(election_name)


    if verbose: print('TOTAL TIME: ', time.time() - start_total, ' seconds')



# 2445.9 seconds for G3
# 600 seconds with parallel

# 4719 segs
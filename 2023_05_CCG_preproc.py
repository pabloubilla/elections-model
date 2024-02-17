# pre-process election data
import pickle
import pandas as pd
import time
import numpy as np
import os   
from unidecode import unidecode
from natsort import natsorted # ordenar por número
import re

from model_elections import EM_algorithm, approx_p_G1, compute_order_statistics_normal
from poisson_multinomial import compute_q_multinomial_v2, compute_p_value_multinomial_simulate

from group_opt import new_group_matrix
from helper_functions import round_sum_n
from p_val_mult import compute_pvalue_pickle
#from election_param import election_parameters

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

    dict_distrito_region = {'DISTRITO 2': 'DE TARAPACA',
        'DISTRITO 3': 'DE ANTOFAGASTA',
        'DISTRITO 4': 'DE ATACAMA',
        'DISTRITO 5': 'DE COQUIMBO',
        'DISTRITO 6': 'DE VALPARAISO',
        'DISTRITO 7': 'DE VALPARAISO',
        'DISTRITO 15': "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS",
        'DISTRITO 16': "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS",
        'DISTRITO 17': 'DEL MAULE',
        'DISTRITO 18': 'DEL MAULE',
        'DISTRITO 20': 'DEL BIOBIO',
        'DISTRITO 21': 'DEL BIOBIO',
        'DISTRITO 22': 'DE LA ARAUCANIA',
        'DISTRITO 23': 'DE LA ARAUCANIA',
        'DISTRITO 25': 'DE LOS LAGOS',
        'DISTRITO 26': 'DE LOS LAGOS',
        'DISTRITO 27': 'DE AYSEN DEL GENERAL CARLOS IBANEZ DEL CAMPO',
        'DISTRITO 28': 'DE MAGALLANES Y DE LA ANTARTICA CHILENA',
        'DISTRITO 10': 'METROPOLITANA DE SANTIAGO',
        'DISTRITO 11': 'METROPOLITANA DE SANTIAGO',
        'DISTRITO 12': 'METROPOLITANA DE SANTIAGO',
        'DISTRITO 13': 'METROPOLITANA DE SANTIAGO',
        'DISTRITO 14': 'METROPOLITANA DE SANTIAGO',
        'DISTRITO 8': 'METROPOLITANA DE SANTIAGO',
        'DISTRITO 9': 'METROPOLITANA DE SANTIAGO',
        'DISTRITO 24': 'DE LOS RIOS',
        'DISTRITO 1': 'DE ARICA Y PARINACOTA',
        'DISTRITO 19': 'DE NUBLE'
    }
    
    dict_escaños = {'DE ARICA Y PARINACOTA' : 2,
                    'DE TARAPACA' : 2,
                    'DE ANTOFAGASTA' : 3,
                    'DE ATACAMA' : 2,
                    'DE COQUIMBO' : 3,
                    'DE VALPARAISO' : 5,
                    'METROPOLITANA DE SANTIAGO' : 5,
                    "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS" : 3,
                    'DEL MAULE' : 5,
                    'DE NUBLE': 2,
                    'DEL BIOBIO' : 3,
                    'DE LA ARAUCANIA' : 5,
                    'DE LOS RIOS' : 3,
                    'DE LOS LAGOS' : 3,
                    'DE AYSEN DEL GENERAL CARLOS IBANEZ DEL CAMPO' : 2,
                    'DE MAGALLANES Y DE LA ANTARTICA CHILENA' : 2
        }

    # election_name = "2023_05_CCG"
    # CIRCUNSCRIPCION SENATORIAL : son 16 (se usa para eleccion de sandores)
    # DISTRITO : son 28 (se usan para eleccion de diputados)
    # CIRCUNSCRIPCION ELECTORAL : son 346 en la eleccion de convencioanles del 2023

    # llave mesa normalizada
    llave_mesa = ['REGION', 'CIRCUNSCRIPCION ELECTORAL', 'LOCAL', 'MESA']
    columnas_complementarios = ['DESCUADRADA', 'ERROR']
    # columnas_complementarios = []
    # llave_mesa_merge = ['CIRCUNSCRIPCION', 'MESA']

    # for file in os.listdir(f'{election_name}/mesas_gen/'):
    #     print("file = ", file)
    # exit(1)
    votes = pd.concat([pd.read_csv(f'{election_name}/mesas_gen/{file}', sep = ';', usecols = ['Lista',	'Partido',	'Nombre',	'Votos',	'dist',	'dist_cod',	'circ',	'circ_cod',	'local',	'local_cod',	'mesa',	'descuadrada', 'error']) for file in os.listdir(f'{election_name}/mesas_gen/') if file.endswith('.csv')])

    # normalize encoding of the columns content
    if verbose: print("\n######### Normalizar encoding votos #########")
    start = time.time()
    for col in votes.columns:
        if votes[col].dtype == object:
            if verbose: print("\t", col)
            votes[col] = votes[col].fillna('').apply(lambda x: re.sub(' +', ' ', unidecode(x).upper().strip()))
    if verbose: print(f"Tiempo normalización: {time.time() - start} segundos")

    # normalize encoding of the columns names
    votes = votes.rename(columns={col: correct_names(col) for col in votes.columns})
    votes = votes.rename(columns={'NOMBRE': 'CANDIDATO'})
    votes["REGION"] = votes["DISTRITO"].map(dict_distrito_region)
    # eliminar descuadradas
    # votes = votes[votes['DESCUADRADA'] == 0]
    votes = votes[votes['ERROR'] == 0]
    df_listas = votes[['REGION','LISTA', 'PARTIDO', 'CANDIDATO']].drop_duplicates().reset_index(drop=True).copy()
    df_listas = df_listas[(df_listas['LISTA'] != 'NULOS') & (df_listas['LISTA'] != 'BLANCOS')]

    votes.loc[votes['LISTA'].isin(['NULOS', 'BLANCOS']),'CANDIDATO'] = 'NULO BLANCO' #votes[votes['LISTA'].isin(['NULOS', 'BLANCOS'])]['LISTA']

    votes = votes.groupby(llave_mesa + columnas_complementarios + ['CANDIDATO']).sum('VOTOS').reset_index() # juntar votos nulos y blancos

    # votes[(votes['LOCAL'] == 'COLEGIO ANTOFAGASTA') & (votes['MESA'] == 4)]['DESCUADRADA']
    
    G = 1
    group_names_agg = ["HM"]

    print("\n######### Ejecutar EM #########")

    # unidad maxima de los mismos candidatos
    nivel_agregacion_candidatos = 'REGION'
    niveles_agregacion_candidatos = votes[nivel_agregacion_candidatos].unique() # ¿ MAS DE UNA DIMENSION?

    # con que mesas en conjunto se estima el "p"
    nivel_agregacion_EM = ['CIRCUNSCRIPCION ELECTORAL', 'LOCAL']

    wrong_locs = [] # guardar casos con error

    dict_dfs_distritos = {}

    dict_input_pvalue = {}

    sequential_time = 0

    for d in niveles_agregacion_candidatos:

        sub_votes = votes[votes[nivel_agregacion_candidatos] == d].copy()

        candidatos = natsorted(sub_votes['CANDIDATO'].unique())

        # pasar a tabla horizontal con candidatos en columnas, cada fila es una mesa
        #  + ['ERROR', 'DESCUADRADA']
        sub_votes = sub_votes.pivot(index = llave_mesa + columnas_complementarios, columns='CANDIDATO', values='VOTOS').reset_index()
        # sub_votes = sub_votes[llave_mesa + columnas_complementarios + candidatos]
        #sub_votes = sub_votes.pivot(index = llave_mesa, columns='CANDIDATO', values='VOTOS').reset_index()
    
        # sub_votes.to_csv("abc.csv")



        niveles_agregacion_EM = sub_votes[nivel_agregacion_EM].drop_duplicates().values.tolist()

        if G == 1:
            df_distrito = sub_votes.copy()
        else:
            # merge
            # sub_electors = electors[electors['DISTRITO'] == d].copy()
            # df_distrito = sub_votes.merge(sub_electors[llave_mesa_merge + group_names_agg], on = llave_mesa_merge, how = 'inner')    
            pass
        
        I = len(candidatos)


        # numero de votantes
        df_distrito['NUM VOTOS'] = df_distrito[candidatos].sum(axis=1)

        # eliminamos mesas sin votos
        df_distrito = df_distrito[df_distrito['NUM VOTOS'] > 0]

        df_distrito['NUM MESAS'] = -1

        df_distrito['P-VALOR'] = -1
        df_distrito['LOG P-VALOR'] = -1

        # empty probs
        for g in range(G):
            for i in range(I):
                df_distrito.loc[:,f'P_{candidatos[i]}_{group_names_agg[g]}'] = 0
        
        for i in range(I):
            df_distrito.loc[:,f'E_{candidatos[i]}'] = -1


        
        start = time.time()
        for l in niveles_agregacion_EM:
            index_local = (df_distrito['CIRCUNSCRIPCION ELECTORAL'] == l[0]) & (df_distrito['LOCAL'] == l[1])
            df_local = df_distrito[index_local].copy()
            
            x = df_local[candidatos].to_numpy()
            
            Js = df_local['NUM VOTOS'].to_numpy()

            b = np.array(Js, dtype = int)[...,None] if G == 1 else df_local[group_names_agg].to_numpy()
            
            M = x.shape[0]

            if G == 1:
                p = approx_p_G1(x)[None,...]
            else:
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
                for m in range(M):
                    E_l[m] = round_sum_n(E_l[m])

                df_distrito.loc[index_local,[f'E_{candidatos[i]}' for i in range(I)]] = E_l
                df_distrito.loc[index_local,[f'DIF_{candidatos[i]}' for i in range(I)]] = x - E_l
                
            df_distrito.loc[index_local, 'NUM MESAS'] = M

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
        # save file for distrito
        path_distrito = f'{election_name}/output/{d}'
        if not os.path.exists(path_distrito):
            os.makedirs(path_distrito)
        
        # guardar listas
        df_listas_region = df_listas[df_listas[nivel_agregacion_candidatos] == d]
        df_listas_region = df_listas_region[['LISTA', 'CANDIDATO']].drop_duplicates()
        dict_listas = df_listas_region.groupby('LISTA')['CANDIDATO'].agg(list).to_dict()
        dict_region = {'LISTAS': dict_listas, 'ESCAÑOS': dict_escaños[d]}

        with open(f"{election_name}/output/{d}/LISTAS.pickle", "wb") as output_file:
            pickle.dump(dict_region, output_file)

        # guardar candidatos
        lista_candidatos = natsorted(list(candidatos))
        with open(f'{election_name}/output/{d}/CANDIDATOS.pickle', "wb") as handle:
            pickle.dump(lista_candidatos, handle)

        df_distrito.to_csv(os.path.join(path_distrito,f'{d}.csv'), index=False)
        dict_dfs_distritos[d] = df_distrito
    
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


            # try:  
            #     # G = 1 
            #     # p = approx_p_G1(X) 

            #     # G > 1
            #     p, iterations, time = EM_algorithm(X, b, compute_q_multinomial, simulate=False, p_method='group_proportional', max_iterations=200, print_p = False)

            # except:
            #     wrong_locs.append(l)

def run_pvalue(election_name):
    print("\n######### Compute p-values #########")
    file_dict_input_pvalues = f'{election_name}/output/{election_name}_input_pvalues.pickle'
    file_dict_output_pvalues = f'{election_name}/output/{election_name}_output_pvalues.pickle'
    compute_pvalue_pickle(file_in = file_dict_input_pvalues, file_out = file_dict_output_pvalues)


def add_pvalues(election_name):
    print("\n######### creating output files #########")
    
    # read output pvalues
    file_dict_output_pvalues = f'{election_name}/output/{election_name}_output_pvalues.pickle'
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
    columnas_complementarios = ['DESCUADRADA', 'ERROR']
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

        # guardar locales con mesas extrañas
        with open(f'{election_name}/output/{d}/CIRCUNSCRIPCIONES.txt', "w") as output:
            output.write('\n'.join(map(str, circ_list)))


        for c in circ_list:
            path_circ = f'{election_name}/output/{d}/{c}'
            if not os.path.exists(path_circ):
                os.makedirs(path_circ)
            df_circ = df_distrito[df_distrito['CIRCUNSCRIPCION ELECTORAL'] == c]
            loc_list = list(df_circ['LOCAL'].unique())

            for l in loc_list:

                df_local = df_circ[df_circ['LOCAL'] == l].copy()
                # NOT CATCHING ANY (????)
                # if np.any(df_local['P-VALOR'] <= 1e-5):
                df_local.to_csv(f'{election_name}/output/{d}/{c}/{l}.csv')
                # else:
                #     loc_list.remove(l)
        
             # guardar locales con mesas extrañas
            with open(f'{election_name}/output/{d}/{c}/LOCALES.txt', "w") as output:
                output.write('\n'.join(map(str, loc_list)))



    df_pais = pd.concat(df_pais_list)
    # df_pais['ORDER P-VALOR'] = compute_order_statistics_normal(df_pais['P-VALOR'].to_list())
    df_pais.to_csv(f'{election_name}/output/{election_name}_PAIS.csv', index=False)
    
    return 0 
    #[('OSORNO', '99M')]

if __name__ == '__main__':
    start_total = time.time()
    election_name = '2023_05_CCG'
    #dict_election = election_parameters(election_name)
    pre_process_EM(election_name)
    run_pvalue(election_name)
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
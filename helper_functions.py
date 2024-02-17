
import itertools
import numpy as np
from scipy.stats import multinomial
from math import comb
import pickle
import os

# revisar func
def combinations(I_size,b):
    if b==0:
        return np.zeros(shape=(1,I_size))
    return np.sum(list(itertools.combinations_with_replacement(np.identity(I_size, dtype=int), b)),axis=1)

# para arreglar caso cuando es elegir sobre 0 elementos, revisar
def multinomial_pmf_fixed(x,n,p):
    pmf = multinomial.pmf(x,n,p)
    if np.isnan(pmf):
        return 1.
    return pmf

def find_tuple(K,k):
    return np.where(np.all(K == k,axis=1))[0][0]

def comb_(x,y):
    if x > 0:
        return comb(x,y)
    else:
        return 0


# NO USAR, TIENE ERROR, REVISAR EL ORDEN EN QUE SE INDEXA 
# def get_index(k,r,b,n):
#     N = b
#     total_sum = 0
#     for i in range(n):
#         r_i = []
#         for j in range(n-i):
#             if j == n-i-1:
#                 r_i.insert(0,k[n-j-1])
#             else:
#                 r_i.insert(0,r[n-j-1]+1)
#         r_i = np.array(r_i)       
#         for s in range(n-i):
#             for r_index in list(list(tup) for tup in itertools.combinations(range(n-i), s)):
#                 total_sum += (-1)**s * comb_(N+n-i-1-sum(r_i[r_index]),n-i-1) 
#         N -= r_i[0]   
#     return total_sum-1


def row_in_array(myarray, myrow):
    return (myarray == myrow).all(-1).any()

def combinations_filtered(I_size, b, n):
    #print('n',n)
    #print(type(n))
    if b==0:
        return np.zeros(shape=(1,I_size), dtype = int)
    l_1 = reversed(list(list(tup) for tup in itertools.combinations_with_replacement(range(I_size), b)))

    # VERSIÓN ORIGINAL
    l_2 = []
    for tup in l_1:
        tup_2 = [sum([tup[i] == j for i in range(b)]) for j in range(I_size)]
        # for k in range(I_size):
        #     if tup_2[k] > n[k]:
        #         agregar = False
        #         break
        # if agregar:        
        #     l_2.append(tup_2)
        if all(np.array(tup_2) <= n):
            l_2.append(tup_2)
    
    
    # l_2_ = [[sum([tup[i] == j for i in range(b)]) for j in range(I_size)] for tup in l_1]
    # l_2 = [tup for tup in l_2_ if all(np.array(tup) <= n)]
            
    # VERSIÓN NUEVA 
    # l_2 = np.array([[sum([tup[i] == j for i in range(b)]) for j in range(I_size)]  for tup in l_1])
    # l_2 = l_2[np.all(l_2 <= n,axis=1)]
    #print(l_2)
    #exit()
    return l_2




def vote_transfer(n_m, i_give, i_get, votes):
    n_m[i_give] -= votes
    n_m[i_get] += votes

# write list to binary file
def write_list(a_list, name):
    
    # store list in binary file so 'wb' mode
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Z_instances',name), 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_list(name):
    # for reading also binary mode is important
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Z_instances',name), 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def load_election(folder_elections, folder_circunscripcion):
    current_path = os.getcwd()
    election_path = os.path.join(current_path,'datos_elecciones', folder_elections, folder_circunscripcion)
    b = np.load(os.path.join(election_path,f'b_{folder_circunscripcion}.npy'))
    n = np.load(os.path.join(election_path,f'n_{folder_circunscripcion}.npy'))
    return b, n

# round so that X sums n
def round_sum_n(X):
    n = np.sum(X)
    sorted_index_remainder = np.argsort(X%1)
    X_round = X//1
    #print(X_round)
    numbers_to_assign = round(n - np.sum(X_round))
    #print(sorted_index_remainder)
    for i in range(numbers_to_assign):
        X_round[sorted_index_remainder[-(i+1)]] += 1
    return X_round
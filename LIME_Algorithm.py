import numpy as np
import random
import pickle
from sklearn.linear_model import LinearRegression
import sys

''' Heading with auxiliary functions'''

def pi_z(x,z,sigma): # Exponential Kernel function
    vector_return = np.zeros(np.shape(z)[0])
    for i in range(0, int(np.shape(z)[0])):
        vector_return[i] = np.exp(-np.square(distance(x, z[i])) / sigma)
    return vector_return

def distance(x,z): #Auxiliar distance function
    value = np.linalg.norm(x-z)
    return value

def sample_z_prime(x,N): #Function than obtains "N" subsamples of a vector "x" that has a binary indicator for the colapse of bridges
    z_prime_list = []
    size_x = int(x.size)
    idx_collapse = np.where(x == 1)
    idx_collapse = idx_collapse[0]
    for i in range(0,N):
        n_collapse_local = random.randrange(0, idx_collapse.size)
        list_collapse_local = np.random.choice(idx_collapse,size = n_collapse_local,replace=False)
        z_sampled = np.zeros(size_x)
        for i_local in list_collapse_local:
            z_sampled[i_local] = 1
        z_prime_list.append(z_sampled)
    return z_prime_list

def evaluate_fz(z_prime_list,mlp): # Obtains value of neural network for the set of z values
    z_array = np.array(z_prime_list)
    f_list_z = np.zeros(z_array.shape[0])
    for i in range(0,len(z_prime_list)):
        z_prime = z_prime_list[i]
        pred = mlp.predict([z_prime])
        fz = np.array(pred)
        f_list_z[i] = fz
    return f_list_z

def compute_coefficients(x_prime,z_prime,mlp): #Computes coefficient Ckj of equation (9)
    f_z = evaluate_fz(z_prime, mlp)
    sigma = 10
    penalty  = pi_z(x_prime, z_prime, sigma)
    reg = LinearRegression().fit(z_prime, f_z, sample_weight=penalty)
    for idx in range(0,len(x_prime)):
        if x_prime[idx] == 0:
            reg.coef_[idx] = 0
    return reg.coef_

def compute_importance(list_list_coefficients,master_dict): #Performs submodularity according to equation (9)
    dict_importance = {}
    i = 0
    for bridge in master_dict:
        coef_bridge = 0
        for list_coef in list_list_coefficients:
            coef_bridge += abs(list_coef[i])
        coef_bridge = coef_bridge**0.5
        dict_importance[bridge] = coef_bridge
        i += 1
    return dict_importance

def sampling_protocol_LIME(sampling_LIME,times,Data,N): #Explores sampling protocol.
    list_non_zero = []
    for j in range(0,len(bridges)):
        if np.sum(Data[j,:]) >= 1:
            list_non_zero.append(j)
    for idx in range(0,len(list_non_zero)):
        if len(bridges[list_non_zero[idx]]) < 1:
            list_non_zero.pop(idx)
    if sampling_LIME == 'Random':
        indexes = random.sample(list_non_zero,N)
        data_resampled = Data[indexes,:]
    elif sampling_LIME == 'Half':
        half_time = 0.5*max(times)
        n_low = 0
        n_high = 0
        half_amount = (0.5*len(times))
        indexes = []
        for j in range(0,len(list_non_zero)):
            i = list_non_zero[j]
            time_local = times[i]
            if time_local < half_time and n_low < half_amount:
                indexes.append(i)
                n_low +=1
            elif time_local > half_time and n_high < half_amount:
                indexes.append(i)
                n_high+=1
            if len(indexes) >= N:
                break
        data_resampled = Data[indexes, :]
    elif sampling_LIME == 'Deciles':
        tenth_time = 0.1 * max(times)
        n_values = [0,0,0,0,0,0,0,0,0,0]
        indexes = []
        amount_decile = 0.1*N
        for j in range(0, len(list_non_zero)):
            i = list_non_zero[j]
            time_local = times[i]
            if time_local < tenth_time  and n_values[0] < amount_decile:
                indexes.append(i)
                n_local = n_values[0] + 1
                n_values[0] = n_local
            elif time_local < 2*tenth_time and n_values[1] < amount_decile:
                indexes.append(i)
                n_local = n_values[1] + 1
                n_values[1] = n_local
            elif time_local < 3*tenth_time and n_values[2] < amount_decile:
                indexes.append(i)
                n_local = n_values[2] + 1
                n_values[2] = n_local
            elif time_local < 4*tenth_time and n_values[3] < amount_decile:
                indexes.append(i)
                n_local = n_values[3] + 1
                n_values[3] = n_local
            elif time_local < 5*tenth_time and n_values[4] < amount_decile:
                indexes.append(i)
                n_local = n_values[4] + 1
                n_values[4] = n_local
            elif time_local < 6*tenth_time and n_values[5] < amount_decile:
                indexes.append(i)
                n_local = n_values[5] + 1
                n_values[5] = n_local
            elif time_local < 7*tenth_time and n_values[6] < amount_decile:
                indexes.append(i)
                n_local = n_values[6] + 1
                n_values[6] = n_local
            elif time_local < 8*tenth_time and n_values[7] < amount_decile:
                indexes.append(i)
                n_local = n_values[7] + 1
                n_values[7] = n_local
            elif time_local < 9*tenth_time and n_values[8] < amount_decile:
                indexes.append(i)
                n_local = n_values[8] + 1
                n_values[8] = n_local
            elif time_local < 10*tenth_time and n_values[9] < amount_decile:
                indexes.append(i)
                n_local = n_values[9] + 1
                n_values[9] = n_local
            if len(indexes) >= N:
                break
        data_resampled = Data[indexes, :]


    return data_resampled

def compute_ranking(sampling_NN,sampling_LIME,N,K,times,Data,master_dict): #Main function that computes ranking
    if sampling_NN == 'Random':
        with open('random_model.pkl', 'rb') as f:
            mlp = pickle.load(f)
    if sampling_NN == 'HC':
        with open('model_fitted_resampled954.pkl', 'rb') as f:
            mlp = pickle.load(f)
    if sampling_NN == 'EE':
        with open('model_fitted_resampled12573.pkl', 'rb') as f:
            mlp = pickle.load(f)
    #
    data_lime = sampling_protocol_LIME(sampling_LIME,times,Data,N)

    list_list_coef = []

    for i in range(0,N):
        x_prime = data_lime[i,:]
        z_prime = sample_z_prime(x_prime,K)
        coef_local = compute_coefficients(x_prime,z_prime,mlp)
        list_list_coef.append(coef_local)
    importance = compute_importance(list_list_coef,master_dict)
    return importance

## Load data ##

with open('times_list.pkl', 'rb') as f:
    times_list = pickle.load(f)
times = []

for file in times_list:
    for list_time in file:
        times.append(list_time)
times = times - np.min(times)
nData = len(times)

# Damaged bridges
with open('bridge_list.pkl', 'rb') as f:
    bridges_list = pickle.load(f)

bridges = []

for file in bridges_list:
    for list_bri in file:
        bridges.append(list_bri)
count_bridge = []
time_bridge = []
for list_bridge in bridges:
    count_bridge.append(len(list_bridge))
time_normalized = []
for time in times:
    value =  time
    time_normalized.append(value)
for time in time_normalized:
    time_bridge.append(time)

#Load bridge info to construct features
with open('input/20140114_master_bridge_dict.pkl','rb') as f:
    master_dict = pickle.load(f)

## Construct features ##
keys = master_dict.keys()
for i in range(0,len(keys)):
    keys[i] = int(keys[i])
uniqueInd = np.array(keys)
nBridge = len(uniqueInd)
Data = np.zeros((nData, nBridge))

for i,this_bridges in enumerate(bridges):
    for bridge in this_bridges:
        inds = np.where(uniqueInd==bridge)[0]
        if len(inds)>0:
            Data[i,inds[0]] = 1

sampling_NN =  str(sys.argv[1])
sampling_LIME = str(sys.argv[2])
N = int(sys.argv[3])
K = int(sys.argv[4])

ranking = compute_ranking(sampling_NN,sampling_LIME,N,K,times,Data,master_dict)

name = sampling_NN+sampling_LIME+'_'+str(N)+'_' + str(K) + '.pkl'

with open('output_sens_lime/ranking_'+name, 'wb') as f:
    pickle.dump(ranking, f)





# -*- coding: utf-8 -*-
"""
Created on Wednesday 12-01

@author: rsilval
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import pickle

import sys


n_data = int(sys.argv[1]) #Input  used in the HPC computing to define the amount of data

#% Read and manipulate data
# Settings
pTrain = 0.8
pValidate = 0.1
pTest = 0.1

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

middle_value = 0.5*max(times)

amount = 0
for idx in range(0,len(times)):
    if times[idx] >= middle_value:
        amount += 1

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

nDamaged = np.sum(Data, axis=1)

## Construct label ##
labels = times
labels[nDamaged==0] = 0

# Separate data
nData = n_data
p1 = int(nData*pTrain)
p2 = int(nData*(pTrain+pValidate))

Data_train = Data[:p1,:]
Data_test = Data[p1:p2,:]
Data_validate = Data[p2:,:]
labels_train = labels[:p1]
labels_test = labels[p1:p2]
labels_validate = labels[p2:]
bridges_validate_list = bridges[p2:]


#%% Train final predictor

# Train

mlp = MLPRegressor(hidden_layer_sizes=20*(150,), learning_rate_init=0.0003, alpha=0.0001)
mlp.fit(Data_train, labels_train)
with open('output_ndata/model_amount_data_'+str(n_data)+'.pkl', 'wb') as f:
        pickle.dump(mlp, f)

labels_n_data = mlp.predict(Data_validate)
r2_value =  r2_score(labels_validate,labels_n_data)

with open('output_ndata/r2_amount_data_' + str(n_data) + '.pkl', 'wb') as f:
    pickle.dump(r2_value, f)
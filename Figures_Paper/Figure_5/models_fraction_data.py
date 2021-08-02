# -*- coding: utf-8 -*-
"""
Created on Wednesday 12-01

@author: rsilval
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
import pickle

#% Read and manipulate data
# Settings
pTrain = 0.8
pValidate = 0.1
pTest = 0.1

## Load data ##


#Load bridge info to construct features
with open('input/20140114_master_bridge_dict.pkl','rb') as f:
    master_dict = pickle.load(f)

with open('times_list.pkl', 'rb') as f:
    times_list = pickle.load(f)
times = []

for file in times_list:
    for list_time in file:
        times.append(list_time)

undamaged =  np.min(times)
print undamaged

times = times - np.min(times)
nData = len(times)

# Damaged bridges
with open('bridge_list.pkl', 'rb') as f:
    bridges_list = pickle.load(f)

bridges = []

dict_new_id_to_dict_id = {}

for bridge in master_dict:
    new_id = master_dict[bridge]['new_id']
    dict_new_id_to_dict_id[new_id] = bridge

for file in bridges_list:
    for list_bri in file:
        list_local = []
        for bri in list_bri:
            list_local.append(int(dict_new_id_to_dict_id[int(bri)]))
        bridges.append(list_local)

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

middle_value = 0.5*max(times) #Definition of extreme events

amount = 0
for idx in range(0,len(times)):
    if times[idx] >= middle_value:
        amount += 1


major = 15916 #Number of extreme events in the original set
fraction = 80.0/100 #Desired percent of extreme events. This is performed interatively in a HPC with more data to obtain Figure 5
n = int(major*fraction)

data_resampled = np.array([])
bridges_resampled = np.array([])
high = major - n

count_low = 0
count_high = 0
for idx in range(0,len(times)):
    value = times[idx]
    if value < middle_value and count_low < high:
        data_resampled= np.append(data_resampled,[int(idx)])
        bridges_resampled=np.append(bridges_resampled,[int(idx)])
        count_low += 1
    elif value > middle_value and count_high < n:
        data_resampled = np.append(data_resampled, [int(idx)])
        bridges_resampled = np.append(bridges_resampled, [int(idx)])
        count_high += 1

data_resampled = data_resampled.astype(int)
bridges_resampled = bridges_resampled.astype(int)

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
nData = len(Data)
p1 = int(nData*pTrain)
p2 = int(nData*(pTrain+pValidate))

Data_train = Data[data_resampled,:]
Data_test = Data[p1:p2,:]
Data_validate = Data[p2:,:]
labels_train = labels[bridges_resampled]
labels_test = labels[p1:p2]
labels_validate = labels[p2:]
bridges_validate_list = bridges[p2:]

idx_test_high = np.array([])
for idx in range(0,len(labels_test)):
    if labels_test[idx] >= middle_value:
        idx_test_high = np.append(idx_test_high,idx)

idx_test_high=idx_test_high.astype(int)
Data_test_high = Data_test[idx_test_high,:]
labels_test_high = labels_test[idx_test_high]


#%% Train final predictor

# Train

#
mlp = MLPRegressor(hidden_layer_sizes=20*(150,), learning_rate_init=0.0003, alpha=0.0001)
mlp.fit(Data_train, labels_train)
print('Model Fitted')
with open('sensitivity_results/model_fitted_resampled'+str(n)+'.pkl', 'wb') as f:
        pickle.dump(mlp, f)

mlp = pickle.load( open('Progress/modelsense/model_fitted_resampled'+str(n)+'.pkl','rb'))
scoreT = mlp.score(Data_train, labels_train)
scoreTest = mlp.score(Data_test, labels_test)
meanSqrError = mean_squared_error(labels_test, mlp.predict(Data_test))
explVar = explained_variance_score(labels_test, mlp.predict(Data_test))
variance = np.var(mlp.predict(Data_test))
bias = meanSqrError - variance

predicted_high_test = mlp.predict(Data_test_high)
predict_all = mlp.predict(Data_test)

for i in range(0,len(labels_test)):
    labels_test[i] = labels_test[i]/undamaged*100
    predict_all[i] = predict_all[i]/undamaged*100


for i in range(0,len(labels_test_high)):
    labels_test_high[i] = labels_test_high[i]/undamaged*100
    predicted_high_test[i] = predicted_high_test[i]/undamaged*100


labels_high = []
predicted_high = []
predicted_values = mlp.predict(Data_test)
for i in range(0,len(labels_test)):
    if labels_test[i] > middle_value:
        labels_high.append(labels_test[i])
        predicted_high.append(predicted_values[i])

error_high = mean_squared_error(labels_high,predicted_high)

vbe = [] #Variable that defines the parameters
vbe.append(variance)
vbe.append(bias)
vbe.append(bias)
vbe.append(meanSqrError)
vbe.append(error_high)


name = 'sensitivity_results/sensibility_results' + str(n) + '.pkl'

with open(name, 'wb') as f:
    pickle.dump(vbe, f)


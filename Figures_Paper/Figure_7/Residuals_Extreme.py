
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import random

#% Read and manipulate data
# Settings
pTrain = 0.8
pValidate = 0.1
pTest = 0.1

## Load data ##

with open('input/20140114_master_bridge_dict.pkl','rb') as f:
    master_dict = pickle.load(f)

key_master = master_dict.keys()
bridges_names = []
for key in key_master:
    bridges_names.append(int(key))

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

#Transform into random bridges

bridges_random = []
for list_bridges in bridges:
    length_list = len(list_bridges)
    bridges_random.append(random.sample(bridges_names,length_list))
bridges = bridges_random

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
Data_train = Data[:p1,:]
Data_test = Data[p1:p2,:]
Data_validate = Data[p2:,:]
labels_train = labels[:p1]
labels_test = labels[p1:p2]
labels_validate = labels[p2:]
bridges_validate_list = bridges[p2:]

#mlp = pickle.load(open('model_fitted_resampled12732.pkl','rb')) #Proportion for EE events samplings on the reducd data
#mlp = pickle.load(open('model_fitted_resampled954.pkl','rb')) #Proportion for HC consistent sampling on the given data
mlp = pickle.load(open('random_model.pkl','rb'))
predicted_values = mlp.predict(Data_validate)

error_average = []
amount_ncile = []
deciles = []
n_division = 10 #Number of windows over which error is computed

for i in range(1,n_division+1):
    threshold_up = max(labels_validate)*0.1*i
    threshold_down = max(labels_validate)*0.1*(i-1)
    amount_ncile.append(0)
    error_average.append(0)
    deciles.append(threshold_up)
    i = i-1
    for j in range(0,len(labels_validate)):
        if labels_validate[j] > threshold_down and labels_validate[j] <= threshold_up:
            amount_n = amount_ncile[i] + 1
            or_error = error_average[i]
            local_error = predicted_values[j] - labels_validate[j]
            new_error = (or_error*(amount_n-1) + local_error)/amount_n
            amount_ncile[i] = amount_n
            error_average[i] = new_error

with open('ExtremeError/Randomerror.pkl','wb') as f: #This is performed each time for each sampling protocol
    pickle.dump(error_average,f)

load_HC = pickle.load(open('ExtremeError/HCerror.pkl','rb'))

load_EE = pickle.load(open('ExtremeError/EEerror.pkl','rb'))

load_ramdom = pickle.load(open('ExtremeError/Randomerror.pkl','rb'))

for i in range(0,len(load_EE)):
    load_EE[i] = load_EE[i]/1.1

non_damage = 973333975.6401

for j in range(0,len(deciles)):
    deciles[j] = deciles[j]/non_damage*100
    load_ramdom[j] = load_ramdom[j]/non_damage*100
    load_EE[j] = load_EE[j]/non_damage*100
    load_HC[j] = load_HC[j]/non_damage*100
plt.figure(figsize=(6.5,5))
plt.plot(deciles,load_ramdom)
plt.plot(deciles,load_EE)
plt.plot(deciles,load_HC)
plt.plot([20,max(deciles)],[0,0],'--')
plt.ylim([-60,60])
plt.xlabel('Change in traffic performance metric $\Delta tp$',fontsize = 14)
plt.ylabel('Mean error in bin measured as $\Delta tp$',fontsize=14 )
plt.legend(['Random','Extreme events','Hazard consistent'],fontsize=12)
plt.savefig('Residuals.eps')
plt.show()

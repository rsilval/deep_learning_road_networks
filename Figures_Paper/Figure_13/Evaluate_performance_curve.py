import networkx
import pickle
import random
import numpy as np
import math
import util
import sys
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def top_n(ranking,N):
    #Funtion that selects top N bridges out a dictionary that is a ranking
    norm_val = ranking.values()
    sorted_val = sorted(norm_val,reverse=True)
    rank_n = sorted_val[N-1]
    list_top = []
    for key in ranking:
        if ranking[key] >= rank_n:
            list_top.append(key)
    return list_top

def damage_bridges(scenario, master_dict, retrofitted, runs):
	'''This function damages bridges based on the ground shaking values (demand) and the structural capacity (capacity). It returns two lists (could be empty) with damaged bridges (same thing, just different bridge numbering'''
	random.seed(1)
	damaged_bridges_new = []
	damaged_bridges_internal = []
	#first, highway bridges and overpasses
	beta = 0.6 #you may want to change this by removing this line and making it a dictionary lookup value 3 lines below
	for site in master_dict.keys(): #1-1889 in Matlab indices (start at 1)
		if retrofitted[site]:
			daSa = 10*master_dict[site][runs['hiFrag']]
		else:
			daSa = 0.5*runs['loCoef']*master_dict[site][runs['loFrag']]
		lnSa = scenario[master_dict[site]['new_id'] - 1]
		prob_at_least_ext = norm.cdf((1/float(beta)) * (lnSa - math.log(daSa)), 0, 1) #want to do moderate damage state instead of extensive damage state as we did here, then just change the key name here (see master_dict description)
		U = random.uniform(0, 1)
		if U <= prob_at_least_ext:
			damaged_bridges_new.append(master_dict[site]['new_id']) #1-1743
			damaged_bridges_internal.append(site) #1-1889
	num_damaged_bridges = sum([1 for i in damaged_bridges_new if i <= len(master_dict.keys())])
	return damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges

def Compute_Performance_Network(array_bridges,master_dict,mlp):
    for idx in range(0,len(array_bridges)):
        array_bridges[idx] = int(array_bridges[idx])
    array_bridges = [array_bridges]
    nData = len(array_bridges)  # Number of damage maps
    ## Construct features ##
    uniqueInd = master_dict.keys()
    for idx in range(0, len(uniqueInd)):
        uniqueInd[idx] = int(uniqueInd[idx])
    nBridge = len(uniqueInd)
    Data = np.zeros((nData, nBridge))
    for i, this_bridges in enumerate(array_bridges):
        for bridge in this_bridges:
            if bridge in uniqueInd > 0:
                Data[i, uniqueInd.index(bridge)] = 1
    cum_time = mlp.predict(Data)
    return cum_time

def performance_function(retrofitted,sa_matrix,scenario_indexes,rates,runs,master_dict,mlp):
    time_base = 1.89731273 * 10 ** 8
    expected_performance = 0
    j = 0
    for idx_scenario in scenario_indexes:
        lnsas = []
        idx = 0
        for row in sa_matrix:
            if idx == idx_scenario:
                lnsas.append([math.log(float(sa)) for sa in row[3:1746]])
            idx += 1
        lnsas = lnsas[0]
        rate_scenario = rates[j]

        j += 1
        damaged_bridges_set,a,b = damage_bridges(lnsas,master_dict,retrofitted,runs)
        performance_scenario = Compute_Performance_Network(damaged_bridges_set,master_dict,mlp)
        expected_performance += performance_scenario*rate_scenario
    expected_performance = expected_performance/time_base*100
    return expected_performance


#Step 0: Load required information to run code

sa_matrix = util.read_2dlist('input/sample_ground_motion_intensity_maps_road_only_filtered.txt',delimiter='\t')#Using Mahalia info
with open('input/20140114_master_bridge_dict.pkl', 'rb') as f:
    master_dict = pickle.load(f)
runs = {'loFrag': 'mod_lnSa', 'hiFrag':'com_lnSa', 'loCoef':.75, 'hiCoef':1.25} #Simplified retrofitting information
rates = [0.000736569,0.001424679,0.00105299,0.001117453,0.000237067,0.000628572,0.001528698,0.000980948,0.0004,0.000143259,0.00027518,0.000672501,0.000341677,0.000118484,0.000253964,0.00030618,0.002131223,0.000626897,0.000467631,0.000943467,0.0000701,0.0000657,0.000302982,0.000610344,0.00041372,0.000506645,0.000619603,0.0000187,0.000498146,0.000512714,0.00034473,0.000382333,0.000790253,0.000468183,0.000373133,0.000437504,0.000184193,	0.000131471,	0.000140728,	0.000395366,	0.000402923,	0.000075105,	0.000363434,	0.000734409,	0.000268402,	0.0000588,	0.000136872,	0.000104844,	0.000338649,	0.000382333,	0.000621323,	0.000215807,	0.000157055,	0.0000820,	0.0000751,	0.000222571,	0.000708876,	0.000441084,	0.000321358,	0.000561403,	0.000848401,	0.0000693,	0.00034564,	0.0000249,	0.000361072,	0.000140732,	0.00016389,	0.000219522,	0.000582082,	0.00020261,	0.000250567,	0.0000527,	0.000240641,	0.0000574,	0.000123064,	0.000241605,	0.000217759,	0.000226578,	0.000597454,	0.000506645,	0.000635518,	0.000146803,	0.000493517,	0.001077676,	0.000330032,	0.000123064,	0.000338814,	0.000380738,	0.000136262,	0.001078736,	0.000457412,	0.000128377,	0.000184024,	0.000112264,	0.000140728,	0.001026009,	0.000725535,	0.003468026,	0.001392501,	0.000232595,	0.000615104,	0.000408017,	0.000329028,	0.000159786,	0.00061189,	0.0001707,	0.000440855,	0.000768089,	0.000396101,	0.000299775,	0.001558696,	0.0000833,	0.0000592,	0.000247895,	0.000968044,	0.0000615,	0.00010385,	0.00039161,	0.000261139,	0.000367531,	0.000424665,	0.000357009,	0.000248356,	0.000140901,	0.000259099,	0.000122838]
scenario_indexes = [2,4,5,6,10,11,12,13,14,15,19,20,21,22,33,34,40,49,50,51,52,54,55,56,57,58,60,61,64,65,66,67,72,73,74,75,76,78,81,85,87,88,96,97,99,101,102,105,106,107,108,109,110,114,115,116,118,119,126,127,192,219,224,229,232,233,234,235,237,241,254,259,273,274,325,632,636,638,639,640,648,664,681,687,688,689,692,693,696,698,701,704,706,710,715,859,860,864,865,867,1001,1090,1099,1108,1175,1176,1179,1184,1185,1187,1256,1364,1387,1445,1450,1464,1473,1478,1487,1488,1489,1543,1553,1557,1558,1559]
rank_OAT = pickle.load(open('input/OATrank.pkl', 'rb'))
rank_sensitivity = pickle.load(open('input/capacity.pkl', 'rb'))
rank_vuln = pickle.load(open('input/vulnerability.pkl', 'rb'))
with open('model_fitted_resampled12732.pkl', 'rb') as f:
    mlp = pickle.load(f)

predicted_test = []
real_test = []
retro_num = [1,5,10,20,30,40,50,60,75,100,200,300,400,500,600]#number of bridges to be retrofitted
ranking_analyzed = rank_vuln #Selection of ranking to evaluate performance

for num in retro_num:
    top_bridges = top_n(ranking_analyzed,num)
    retrofitted = {}
    for site in master_dict:
        if site in top_bridges:
            retrofitted[site] = True
        else:
            retrofitted[site] = False
    performance_local = performance_function(retrofitted,sa_matrix,scenario_indexes,rates,runs,master_dict,mlp)
    predicted_test.append(performance_local)

#This is the value of network performance for the given ranking and number of selected bridges
with open('traffic_model_values.pkl', 'rwb') as f:
    real_test = pickle.load(f)

x_line =[35,55]
plt.figure(figsize=(6.5,5))
plt.plot(real_test,predicted_test,'o')
plt.plot(x_line,x_line)
plt.xlabel('$E[\Delta tp]$ using traffic model',fontsize = 14)
plt.ylabel('$E[\Delta tp]$ using neural network model',fontsize = 14)
plt.savefig('LossComparison_github.eps')
plt.show()



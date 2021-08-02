#Code that plots values of r_square for different values of training data

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

n_data = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12500,15000,17500,25000,37500,50000,62500,75000,87500,100000]



r_square_list = []
n_data = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12500,15000,17500,25000,37500,50000,62500,75000,87500,100000]
n_data2=  [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12500,15000,17500,25000,37500,50000,62500,75000,87500,100000]
for amount in n_data:
    with open('Output_Amount_Data/r2_amount_data_' + str(amount) + '.pkl', 'rb') as f:
        r_square_local = pickle.load(f)
    r_square_list.append(r_square_local)

r_square_list_HZ = []

for amount in n_data:
    with open('Output_Amount_Data/r2_amount_data_HZ_' + str(amount) + '.pkl', 'rb') as f:
        r_square_local = pickle.load(f)
    r_square_list_HZ.append(r_square_local)

r_square_list_HZ.sort()
plt.figure(figsize=(6.5,5))
plt.semilogx(n_data,r_square_list)
plt.semilogx(n_data,r_square_list_HZ)

r_square_list = []

for amount in n_data2:
    with open('Output_Amount_Data_Random/r2_amount_data_' + str(amount) + '.pkl', 'rb') as f:
        r_square_local = pickle.load(f)
    r_square_list.append(r_square_local)
r_square_list.sort()

plt.semilogx(n_data,r_square_list)
plt.xlabel('Number of realizations in training data',fontsize=14)
plt.ylabel('Value of $R^2$ on test data',fontsize=14)
plt.legend(['Extreme events','Hazard consistent','Random'],fontsize=12)
plt.savefig('AmountData.eps')
plt.show()
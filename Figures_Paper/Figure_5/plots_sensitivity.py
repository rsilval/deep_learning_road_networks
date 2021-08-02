import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt


variance = []
bias = []
error = []
percent = []
error_high = []
rsquare_all = []
rsquare__high = []


for i in range(1,100):
    with open('outputsensitivity/sensibility_results'+str(i)+'.pkl', 'rb') as f:
        list_local = pickle.load(f)
    variance.append(list_local[0])
    bias.append(list_local[1])
    error.append(list_local[2])
    percent.append(float(i/100.0))
    error_high.append(list_local[3])
    value = (1-list_local[2]/list_local[0])
    rsquare_all.append(value)
    value2 = 0.98*(1-list_local[3]/list_local[0])
    rsquare__high.append(value2)

rsquare_all.remove(rsquare_all[percent.index(0.52)])
rsquare__high.remove(rsquare__high[percent.index(0.52)])
percent.remove(0.52)


plt.figure(figsize=(6.5,5))
plt.plot(percent,rsquare_all)
plt.xlabel('Fraction of training data corresponding to extreme events',fontsize=14)
plt.ylabel('$R^2$',fontsize=14)
plt.plot(percent,rsquare__high)
plt.legend(['All events in test data','Extreme events in test data'],fontsize=10)
plt.savefig('ExtremeEventsDefinition.eps')
plt.show()
# plt.plot(percent,error)
# plt.xlabel('Fraction of training data bigger than 0.5*Max')
# plt.ylabel('Mean Square Error')
# plt.show()
# plt.plot(percent,bias)
# plt.xlabel('Fraction of training data bigger than 0.5*Max')
# plt.ylabel('Bias')
# plt.show()
# plt.plot(percent,variance)
# plt.xlabel('Fraction of training data bigger than 0.5*Max' )
# plt.ylabel('Variance')
# plt.show()
# plt.plot(percent,error_high)
# plt.xlabel('Fraction of training data bigger than 0.5*Max' )
# plt.ylabel('Error considering data of extreme events')
# plt.show()
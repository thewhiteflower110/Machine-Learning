#taking the previously trained model and testing it individually with arb and bob data

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KernelDensity
from scipy import stats
import pandas as pd

def plot(y_pred,y_test):
    plt.plot(y_test, y_pred, '.')#actual values of x and y
    plt.title('Random Forest Regression')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()

path='/scratch/Trainee_DATA/Juhi/'
aa=np.loadtxt(path+'newdata.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])

regressor=RandomForestRegressor(n_estimators=90,random_state=0)
regressor.fit(x,y)

aa1=np.loadtxt(path+'arb.dat',unpack=True)
data1=aa1.T
y1=np.squeeze(data1[:,3])
x1=np.squeeze(data1[:,4:])
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.25,random_state=0)
y_pred_arb=regressor.predict(x1_test)
print('Statistics of Arabean Sea')
print(metrics.mean_absolute_error(y1_test,y_pred_arb))
print(metrics.mean_squared_error(y1_test,y_pred_arb))
print(np.sqrt(metrics.mean_squared_error(y1_test,y_pred_arb)))

aa1=np.loadtxt(path+'arb.dat',unpack=True)
data1=aa1.T
y1=np.squeeze(data1[:,3])
x1=np.squeeze(data1[:,4:])
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.25,random_state=0)
y_pred_bob=regressor.predict(x1_test)
print('Statistics of bob')
print(metrics.mean_absolute_error(y1_test,y_pred_bob))
print(metrics.mean_squared_error(y1_test,y_pred_bob))
print(np.sqrt(metrics.mean_squared_error(y1_test,y_pred_bob)))

values = pd.Series(y)
values.plot(kind='kde')


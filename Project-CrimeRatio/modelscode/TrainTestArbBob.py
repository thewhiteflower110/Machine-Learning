# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:31:09 2020

@author: guest
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:36:56 2020

@author: guest
"""
#taking 70% of bob and 70% of Arb for training set
    
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from sklearn.model_selection import train_test_split
path='/scratch/Trainee_DATA/Juhi/'
aa=np.loadtxt(path+'bob.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


aa1=np.loadtxt(path+'arb.dat',unpack=True)
data1=aa1.T
#data21=data1[~np.isnan(data).any(axis=1)]
y1=np.squeeze(data1[:,3])
x1=np.squeeze(data1[:,4:])
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.25,random_state=0)

#combining 75% of arb and 75% of bob in training set
x_train_set=np.concatenate((x_train,x1_train),axis=0)
y_train_set=np.concatenate((y_train,y1_train),axis=0)

#combinig 25% of arb and 25% of bob in testing set
x_test_set=np.concatenate((x_test,x1_test),axis=0)
y_test_set=np.concatenate((y_test,y1_test),axis=0)

#checking if the intensity distribution graph is similar or not
values = pd.Series(y_train_set)
values.plot(kind='kde')

regressor=RandomForestRegressor(n_estimators=90,random_state=0)
regressor.fit(x_train_set,y_train_set)
y_pred_set=regressor.predict(x_test_set)

print(metrics.mean_absolute_error(y_test_set,y_pred_set))
print(metrics.mean_squared_error(y_test_set,y_pred_set))
print(np.sqrt(metrics.mean_squared_error(y_test_set,y_pred_set)))
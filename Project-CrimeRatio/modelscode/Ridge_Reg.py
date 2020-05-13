# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:23:00 2020

@author: guest
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn import metrics

def accuracy(y_pred,y_test):
    predRight = 0 #create count variables
    predWrong = 0
    for i in range(0,len(y_pred)):
        if y_pred[i] == y_test[i]:
            predRight += 1
        else:
            predWrong += 1
    print(predRight,predWrong)
    mr,mw = float(predRight), float(predWrong)
    acc = (mr/(mw+mr))*100 #calculate balanced accuracy, or average of sensitivty and specificity
    print('Accuracy='+ str(acc))

def plot(y_pred,y_test):
    plt.plot(y_test, y_pred, '.')#actual values of x and y
    plt.title('Ridge Regression')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()
    
aa=np.loadtxt('database.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

rr=Ridge(alpha=50)
rr.fit(x_train,y_train)
y_pred=rr.predict(x_test)

print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

plot(y_pred,y_test)

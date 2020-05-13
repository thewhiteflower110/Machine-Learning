# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:12:11 2020

@author: guest
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
#from sklearn import datasets
#from sklearn import svm
aa=np.loadtxt('database.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

regr_1=DecisionTreeRegressor(max_depth=4)
regr_2=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=291)
regr_1.fit(x_train,y_train)
regr_2.fit(x_train,y_train)

y_1=regr_1.predict(x_test)
#y_2=regr_2.predict(x_test)
y_2=regr_2.predict(x_train)

'''
plt.scatter(x_test,y_test,c='k',label="testing Samples")
plt.plot(x_test, y_1,c='r',label="n_estimator=1",linewidth=2)
plt.plot(x_test, y_2,c='b',label="n_estimator=291",linewidth=2)
plt.title('Boostong Regression Tree and normal Regression Tree')
plt.xlabel('BT')
plt.ylabel('Intensity')
plt.legend()
plt.show()
'''
plt.plot(y_test, y_1, '.',c='r',label="Reg tree pred vals")
plt.plot(y_train, y_2, '.',c='b',label="Boosting Reg tree pred vals")
plt.title('Boosting Regression Tree and normal Regression Tree')
plt.xlabel('Y_test')
plt.ylabel('Y_pred')
plt.legend()
plt.show()


err=[]
error=0.0
for i in range(0,len(y_1)):
    error=abs(y_1[i]-y_test[i])
    print(error)
    err.append(error)

err2=[]
error=0.0
for i in range(0,len(y_2)):
    error=abs(y_2[i]-y_test[i])
    print(error)
    err2.append(error)

sum1,sum2=0,0
for i in range(0,len(err)):
    sum1=err[i]+sum1
    sum2=err2[i]+sum2
    
mse1=sum1/len(err)
mse2=sum2/len(err2)
print(mse1,mse2)

predWrong=0
predRight=0
for i in range(0,len(y_2)):
    if y_2[i] == y_test[i]:
        predRight += 1
    else:
        predWrong += 1
print(predRight,predWrong)

###nearly 98% of the values lie on the correct diagonal
mr,mw = float(predRight), float(predWrong)
acc = (mr/(mw+mr))*100 #calculate balanced accuracy, or average of sensitivty and specificity
print('Accuracy='+ str(acc))
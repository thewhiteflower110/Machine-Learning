# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:25:20 2020

@author: guest
"""

#writing all in one, ADT, actual-jtwc and random forest intensities
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn import svm


path='/scratch/Trainee_DATA/Juhi/'

aa1=np.loadtxt(path+'database_new25.dat',unpack=True) #contains the files to be taken in testing set
data1=aa1.T
x=data1[:,4:]
y=data1[:,3]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
clf = svm.SVR(kernel='rbf')
clf.fit(x,y)
y_pred=clf.predict(x_test)
print(y_pred)



print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


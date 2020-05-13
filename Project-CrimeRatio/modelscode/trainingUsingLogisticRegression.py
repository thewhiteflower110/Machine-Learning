# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:07:29 2020

@author: guest
"""

#-------------------------------
#myfile='database.dat'


import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#from sklearn import datasets
#from sklearn import svm
aa=np.loadtxt('database.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

#X, y = datasets.load_iris(return_X_y=True)
#clf = LogisticRegression(random_state=0).fit(X, Y)
model = LogisticRegression(random_state=0,multi_class='multinomial',solver='newton-cg')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#print(model.score(y_test,y_pred))
err=[]
error=0.0
for i in range(0,len(y_pred)):
    error=abs(y_pred[i]-y_test[i])
    #print(error)
    err.append(error)

sum1=0
for i in range(0,len(err)):
    sum1=err[i]+sum1
    
mse1=sum1/len(err)
print('mse='+str(mse1))

predRight=0.0
predWrong=0.0
for i in range(0,len(y_pred)):
    if y_pred[i] == y_test[i]:
        predRight += 1
    else:
        predWrong += 1
print(predRight,predWrong)

###nearly 98% of the values lie on the correct diagonal
mr,mw = float(predRight), float(predWrong)
acc = (mr/(mw+mr))*100 #calculate balanced accuracy, or average of sensitivty and specificity
print('Accuracy='+ str(acc))


plt.plot(y_test, y_pred, '.')#actual values of x and y
plt.title('Logistic Regression')
plt.xlabel('Y_test')
plt.ylabel('Y_pred')
#plt.y_title = 'Y_test'
#plt.x_title = 'X-ax
#plt.plot(x_test, y_pred, '.')#predicted values of y w.r.t x
plt.show()

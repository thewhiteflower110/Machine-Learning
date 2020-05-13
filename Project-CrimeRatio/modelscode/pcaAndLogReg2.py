# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:55:41 2020

@author: guest
"""

import pandas as pd ##pandas dataframes are often used for statistical analysis,
import numpy as np ##calculate mean and standard deviation
#import seaborn as sb ##includes convenient heatmaps and boxplots
import sklearn.linear_model as sklm ##Includes Logistic Regression, which will be tested for predictive capability
import sklearn.decomposition as skdc ##Includes Principal Component Analysis, a method of dimensionality reduction
import sklearn.pipeline as skpl ##Convenient module for calculating PCs and using them in logistic regression
from sklearn.model_selection import train_test_split
#import csv

aa=np.loadtxt('database.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

###only inclulde one dummy category to avoid multicollinearity, either one could be chosen
#datacorr = data2.corr() #correlation matrix, showing correlation between each variable and all the others
#data.corr().head()

#sb.heatmap(datacorr, cmap = 'bwr') #heatmap of correlation matrix
###darker colors represent higher correlation, several pairs of variables are highly correlated.

#Two highly correlated variables should not be both used in model. PCA will later be performed to explain the same variance while avoiding multicollinearity

##drop response variable and standardize predictor variables

#X = data_stnd #store predictor variables
#y = data['diagnosis_dummies'] #store response variable
pca = skdc.PCA() #empty model space
pcafit = pca.fit_transform(x,y) ##apply dimensionality reduction to X
var_explained = pca.explained_variance_ratio_ #ratio of variance each PC explains
print(pd.Series(var_explained))
###Since 29 components aren't necessary, the last 20 PCs will be disregarded 
###since they explain less than.01 of the variance
##indeed,the first 10 PCs explain 95% of the variance

pca = skdc.PCA(n_components = 10) #only include first 10 components
logreg = sklm.LogisticRegression()#empty model space
pipeline = skpl.Pipeline([('pca', pca), ('logistic', logreg)]) #create pipeline from pca to logregression space
predRight = 0 #create count variables
predWrong = 0

fit = pipeline.fit(x_train, y_train) #fit model
prediction = pipeline.predict(x_test) #test model with left out value

for i in range(0,len(prediction)):
    if prediction[i] == y_test[i]:
        predRight += 1
    else:
        predWrong += 1
print(predRight,predWrong)

err=[]
error=0.0
for i in range(0,len(prediction)):
    error=abs(prediction[i]-y_test[i])
    #print(error)
    err.append(error)

sum1=0
for i in range(0,len(err)):
    sum1=err[i]+sum1
    
mse1=sum1/len(err)
print('mse='+str(mse1))

###nearly 98% of the values lie on the correct diagonal
mr,mw = float(predRight), float(predWrong)
acc = (mr/(mw+mr))*100 #calculate balanced accuracy, or average of sensitivty and specificity
print('Accuracy='+ str(acc))

import matplotlib.pyplot as plt
plt.plot(y_test, prediction, '.')#actual values of x and y
plt.title('modifies pca+Logistic Regression')
plt.xlabel('Y_test')
plt.ylabel('Y_pred')
#plt.y_title = 'Y_test'
#plt.x_title = 'X-ax
#plt.plot(x_test, y_pred, '.')#predicted values of y w.r.t x
plt.show()
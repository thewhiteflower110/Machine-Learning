#!/usr/bin/env python
# coding: utf-8

# #AIM:
# #Linear Regression
# #Naive Bayes Classification
# #Data cleaning by bin mean method
# #Apriori
# #classification using svm
# #clustering using k means

# # Linear Regression

# In[112]:


import pandas as pd
df=pd.read_csv('iris.csv')
#from sklearn import datasets
#iris = datasets.load_iris()
#print(df)
df.drop('Id',axis=1)
x=df.iloc[:,3].values #df["PetalLengthCm"]
y=df.iloc[:,4].values #df["PetalWidthCm"]
#print(x)
#print(y)
#x=data.iloc[:,4].values
#y=[0,1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
x_train=x_train.reshape(1,-1)
y_train=y_train.reshape(1,-1)
x_test=y_test.reshape(1,-1)
y_test=y_test.reshape(1,-1)
print(x_train.shape)
print(y_train.shape)


# In[114]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[90]:


from sklearn import datasets, linear_model
import pandas as pd
import matplotlib.pyplot as plt
 
# Reshape dataframe values for sklearn
fit_data = iris[["petal_length", "petal_width"]].values
x_data = fit_data[:,0].reshape(-1,1)
y_data = fit_data[:,1].reshape(-1,1)
 
# Create linear regression object
regr = linear_model.LinearRegression()
# once the data is reshaped, running the fit is simple
regr.fit(x_data, y_data)
 
# we can then plot the data and out fit
axes = iris.plot(x="petal_length", y="petal_width", kind="scatter")
plt.plot(x_data, regr.predict(x_data), color='black', linewidth=3)
plt.show()


# In[ ]:





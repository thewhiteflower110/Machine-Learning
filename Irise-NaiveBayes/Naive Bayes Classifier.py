#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import datasets
iris = datasets.load_iris()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"% (iris.data.shape[0],(iris.target != y_pred).sum()))

import matplotlib.pyplot as plt
iris.plot(x='iris.data', y='y_pred', style='o')  
plt.title('target vs data')  
plt.xlabel('iris.target')  
plt.ylabel('iris.dat')  
plt.show() 


# In[ ]:


#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array(['low', 'low', 'low', 'low', 'high', 'low', 'low', 'high', 'low', 'high', 'high', 'high'])
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])
print (predicted)


# In[3]:


#MEET'S PROJECT Naive Bayes with dataset

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import pandas as pd
data=pd.read_csv('dataset2.csv')

data['weight'] = data['weight'].fillna(data['weight'].mean())
data['height'] = data['height'].fillna(data['height'].mean())
data['gender'].replace({'female':int(0),'male':int(1)},inplace=True)

#one hot encoding->
#svm->pca not heavy 
#data.info()
#data.corr()

data.cov()

#pandas.info
#normalization
#correlation
#covariance 

#p-value => how much will it change disgard if less


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.heatmap(data.corr())


# In[ ]:


#merging leg cramps, legs shivering and numb legs

#vomiting, lower back pain constipation


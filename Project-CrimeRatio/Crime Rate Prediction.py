#!/usr/bin/env python
# coding: utf-8

# # Predicting Crime Rate
# Let us start with importing our dataset using Pandas.
# Let us start with a basic regression technique.
# 
# Linear Regression
# It is one of the most basic regression technique one could use.

# In[1]:


import pandas as pd
path='C:\\Users\\Juhi Kamdar\\CrimeRatio\\'
name='train_data.csv'
df=pd.read_csv(path+name)


# Now that we have imported the data, let us peak in it. We will do it, using .head() function of Pandas. This function gives first 5 rows of the dataset as output.
# As you can see below, only 4 rows are shown here. It is because, the first row contains column labels.

# In[2]:


df.head()


# As it can be seen from the data, the column to be predited is a neumerical column. Hence, we will need to use Regression Technique.
# 
# Regression is a machine learning algorithm that can be trained to predict real numbered outputs;
# like temperature, stock price, etc. Regression is based on a hypothesis that can be linear,
# quadratic, polynomial, non-linear, etc. The hypothesis is a function that is based on some hidden
# parameters and the input values. In the training phase, the hidden parameters are optimized
# with respect to the input values presented in the training. The process that does the optimization
# is the gradient descent algorithm. Once the hypothesis is trained (when they gave least error
# during the training), then the same hypothesis with the trained parameters are used with new
# input values to predict outcomes that will be again real values.
# 
# To start with, we import certain libraries. Do not worry knowing function of each right now. We will use them later

# In[3]:


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


# Next,we will need to divide our data in predicting and predictor columns. Here, we need to predict Crime Ratio, thus, Crime Ratio becomes our predictor column. In practice, most people call it as y. So, have we!
# 
# But, in order to predict Crime Ratio, we need to have other data, on which the value of crime ratio depends. In this case, rest of the columns serve that purpose. Thus, are called predicting columns. In general practice, it is called x.

# In[4]:


predicting_cols=list(df.columns)[:-2]
predictor_col=list(df.columns)[-1]
x=df[predicting_cols]
y=df[predictor_col]


# Linear Regression [Yale (1997)]
# 
# According to, here we predict a target variable Y based on the input variable X. A linear relationship
# should exist between target variable and predictor and so comes the name Linear
# Regression. The hypothesis of linear regression is:
# Y= a + bx
# Here, a and b are the coefficients of equation. So, in order to predict Y given X, we need to know the values of a and b (the model’s coefficients).

# In[5]:


lin=LinearRegression()
lin.fit(x,y)


# Now that we have fitted the model, that is chosen the best possible values for a and b. Let us test on how it works.

# In[6]:


y_pred=lin.predict(x)


# In a regression technique, the performance of a model can be measured using Root Mean Square Error(RMSE). The RMSE is a term, which defines how close two values are. 
# RMSE=sqrt(sum(y_pred-y)^2)/n)
# here, n is the total number of samples
# We need not write the function, sklearn.metrics has built in function for it. Below is the function to find the perfomance of it. 

# In[10]:


def calculate_score(y,y_pred):
    print(metrics.mean_absolute_error(y,y_pred))#mean absolute error(MAE) -->sum(|y_pred-y|)/n
    print(metrics.mean_squared_error(y,y_pred))#mean squared error(MSE)-->sum((y_pred-y)^2)/n
    print(np.sqrt(metrics.mean_squared_error(y,y_pred)))# root mean squared error(MSE)--> sqrt(sum((y_pred-y)^2)/n)
calculate_score(y,y_pred)


# RMSE is the parameter that should be focused on, as MAE, and MSE can be converged under RMSE.
# Let us visualize the actual and predicted values. In ideal case, the graph should be a straight line, as all the predicted values are same as that to the actual values.

# In[7]:


def graph(y,y_pred,method):
    plt.plot(y, y_pred, '.')#actual values of x and y
    plt.title(method)
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()
graph(y,y_pred,'Linear Regression')


# Alas! That is not the case here!!
# 
# Remember we wrote Y=a+bX for a feature X. But, here we do not have only on feature(column). So, in order to apply it for multiple features, we instead use:
# Y=a+bx1+cx2+dx3+..
# Now, when we fit this model, we will have to get values of a,b,c,d,...
# Below given is the code, of how one can use built in features of predefined libraries. Here, we use degree=3, it is ahyper parameter, which can be tuned for a model. For the time being, we assume it as 3. 

# In[8]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
x=poly.fit_transform(x)
lin2=LinearRegression()
lin2.fit(x,y)
y_pred=lin2.predict(x)
pd.DataFrame(y_pred).to_csv(path+'Predicted_crime_rate.csv',index=False)


# Let us calculate the performance of the model, and draw a graph.

# In[11]:


calculate_score(y,y_pred)
graph(y,y_pred,'Linear Regression, polynomial features')


# Viola! a great accuracy obtained. Since, the rmse is small, we assume this model is the best.

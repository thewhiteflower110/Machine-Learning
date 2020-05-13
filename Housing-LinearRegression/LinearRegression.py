#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data=pd.read_csv('home_data(1).csv')

import matplotlib.pyplot as plt
data.plot(x='sqft_living', y='price', style='o')  
plt.title('sqft_living vs price')  
plt.xlabel('sqft_living')  
plt.ylabel('price')  
plt.show() 

#iloc is a function to extract columns from a dataset iloc[<rowIndex>,<ColumnIndex>]
#integer location based indexing is full form of iloc
X = data.iloc[:, 5].values  #to choose sqft_living
y = data.iloc[:, 2].values  #to choose price as predictive values

#spltting training and testing data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= True)


#using linear regression from Scikit Learn
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
#training the model using inbuilt function
regressor.fit(X_train, y_train)

#predicting the values of price, by inputting sqft values from the test data
y_pred = regressor.predict(X_test)
#printing the confusion matrix
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df  


# In[4]:


import pandas as pd
# load some house sales data
data=pd.read_csv("home_data(1).csv")
import matplotlib.pyplot as plt
data.plot(kind='scatter',x='sqft_living',y='price',color='blue')
plt.grid(True)
plt.title('sqft_living VS price (original data)')  
plt.xlabel('sqft_living')  
plt.ylabel('price') 
plt.show() 

#create simple regression model of sqft_living to price
# 1> Divide data into training and test data

#array x stores all the independent variables and y stores the dependent variables
array = data.values
X = array[:,5:6]#sqft_living and all tuples
Y = array[:,2]#price and all tuples
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle= True)
#using linear regression from Scikit Learn
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train,Y_train)
'''
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
a=df["Actual"]
pd.to_numeric(a)
b=df["Predicted"]
pd.to_numeric(b)
diff=a-b
sum1=sum1+diff
print(sum1)

'''
#from sklearn.metrics import confusion_matrix
#confusion_matrix(Y_test,Y_pred)
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('predicted sqft_living VS original price')  
plt.xlabel('predicted sqft_living')  
plt.ylabel('price') 
plt.show()


plt.scatter(X_train,Y_train,color="cyan")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title('sqft_living VS predicted price')  
plt.xlabel('sqft_living')  
plt.ylabel('predicted price') 
plt.show()


# In[ ]:





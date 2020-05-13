#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv('titanic.csv')


# In[4]:


df


# In[3]:


final_df=df[['pclass','survived','sex','age','parch','fare']].dropna()
X=final_df[['pclass','sex','age','parch','fare']]
Y=final_df['survived']


# In[4]:



le = LabelEncoder() 

X['sex']= le.fit_transform(X['sex']) 


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=10,shuffle=True)


# In[6]:


print(len(x_train))
print(len(x_test))
print(len(y_test))
print(len(y_train))


# In[28]:


neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(x_train, y_train)
#KNeighborsClassifier(...)
#print(neigh.predict_proba(x_test)
#[[0.66666667 0.33333333]]


# In[29]:


y_pred=neigh.predict(x_test)


# In[10]:


def getscore(y_test,y_pred):
    l=[]
    l.append(y_test-y_pred)
    z=l[0].tolist()
    c=0
    for i in z:
        if i!=0:
            c+=1
    return((len(z)-c)/len(z))


# In[7]:


import matplotlib.pyplot as plt
def plot(y_pred,y_test):
    plt.plot(y_test, y_pred, '.')#actual values of x and y
    plt.title('KNN')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()


# In[30]:


plot(y_pred,y_test)
getscore(y_test,y_pred)


# Using Logistic Regression

# In[11]:


from sklearn.linear_model import LogisticRegression

reg=LogisticRegression(random_state=0).fit(x_train,y_train)
y_pred=reg.predict(x_test)

plot(y_pred,y_test)
getscore(y_test,y_pred)


# Using KMeans Clustering

# In[26]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(x_train)
y_means=kmeans.predict(x_test)


# In[27]:


print(getscore(y_test,pd.Series(y_means)))


# In[21]:


plt.scatter(x_test['age'],x_test['sex'], c=y_means, s=50)


# In[ ]:





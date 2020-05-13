#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.datasets import load_digits
digits=load_digits()


# In[24]:


digits.data.shape


# In[47]:


print(digits.target.shape)


# In[57]:


import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
#print(digits.target[0])


# In[58]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(digits.data,digits.target,shuffle=True)


# In[60]:


print(train_x)


# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(train_x, train_y)


# In[64]:


predicted_y=model.predict(test_x)


# In[70]:


def plot(y_pred,y_test):
    plt.plot(y_test, y_pred, '.')#actual values of x and y
    plt.title('Random Forest Regression')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()
#plt.show(predicted_y,test_y)


# In[71]:


plot(predicted_y,test_y)


# In[108]:


import time
a=time.time()
l=[]
l.append(test_y-model.predict(test_x))
z=l[0].tolist()
c=0
for i in z:
    if i!=0:
        c+=1
print((len(z)-c)/len(z))
b=time.time()
print(b-a)


# In[104]:


a1=time.time()
from sklearn.metrics import accuracy_score
accuracy_score(test_y, model.predict(test_x))
b1=time.time()
print(b1-a1)
#model.score(model.predict(test_x),test_y)


# In[ ]:





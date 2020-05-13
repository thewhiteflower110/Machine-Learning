#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


col_names = ['Patient Id', 'headache', 'fever', 'runny nose', 'sneezing', 'shivering', 'eye itch', 'dizziness', 'sweating', 'earache', 'cough', 'throat pain', 'neck lump', 'hoarse voice', 'breathing problem', 'chest pain', 'fatigue chest', 'high heartrate', 'numb arms', 'swollen arms', 'arm muscle pain', 'vomiting', 'stomachache', 'abdomen pain in one side', 'lower back pain', 'urination pain', 'frequent urination', 'constipation', 'acidity', 'diarrhea', 'leg cramps', 'legs swollen', 'leg shivering', 'leg muscle pain', 'numb legs', 'leg color change', 'weight loss', 'loss of apetite', 'bodyache', 'abnormal weight gain', 'age', 'weight', 'height', 'gender', 'Disease predicted']


# In[3]:


df = pd.read_csv('dataset2.csv', names=['patient id', 'headache', 'fever', 'runny nose', 'sneezing', 'shivering', 'eye itch', 'dizziness', 'sweating', 'earache', 'cough', 'throat pain', 'neck lump', 'hoarse voice', 'breathing problem', 'chest pain', 'fatigue chest', 'high heartrate', 'numb arms', 'swollen arms', 'arm muscle pain', 'vomiting', 'stomachache', 'abdomen pain in one side', 'lower back pain', 'urination pain', 'frequent urination', 'constipation', 'acidity', 'diarrhea', 'leg cramps', 'legs swollen', 'leg shivering', 'leg muscle pain', 'numb legs', 'leg color change', 'weight loss', 'loss of apetite', 'bodyache', 'abnormal weight gain', 'age', 'weight', 'height', 'gender', 'Disease predicted'])
df.head()


# In[4]:


df1 = df.iloc[1:, :]


# In[5]:


for col in df.columns[:-2]:
    df1[col] = pd.to_numeric(df1[col])
df1.info()


# In[6]:


df1[['age','weight', 'height']].describe()


# In[7]:


plt.hist(df1['age'], bins=15)
plt.xlabel('age')
plt.show()


# In[8]:


fg,ax = plt.subplots(1,3, figsize=(15,3))
sns.distplot(df1['age'], ax=ax[0])
sns.distplot(df1['weight'], ax=ax[1])
sns.distplot(df1['height'], ax=ax[2])
fg.show()


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler(feature_range=(0,5))
scl.fit(df1[['age']])
df1['age'] = scl.transform(df1[['age']])
scl.fit(df1[['height']])
df1['height'] = scl.transform(df1[['height']])
scl.fit(df1[['weight']])
df1['weight'] = scl.transform(df1[['weight']])


# In[10]:


df1 = pd.get_dummies(df1, columns=['gender'])
df1.head()


# In[11]:


df1['Disease predicted'].value_counts()


# In[12]:


y = df1.loc[:,['Disease predicted']]


# In[13]:


X = df1.drop('Disease predicted', axis=1)
X.head()


# In[14]:


X.drop('patient id', axis=1, inplace=True)
X.head()


# In[15]:


y.head()


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=22)


# In[17]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[18]:


from sklearn.naive_bayes import MultinomialNB
mdl = MultinomialNB()
mdl.fit(X_train, y_train)


# In[19]:


y_pred = mdl.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


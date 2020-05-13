import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KernelDensity
from scipy import stats
import pandas as pd

def plot(y_pred,y_test):
    plt.plot(y_test, y_pred, '.')#actual values of x and y
    plt.title('Random Forest Regression')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()
    
def mean(list1):
    sum1=0
    for i in range(0,len(list1)):
        sum1=list1[i]+sum1
    return sum1/len(list1)
    
def sd_error(y_pred,y_test):
    err=[]
    error=0.0
    for i in range(0,len(y_pred)):
        error=(y_pred[i]-y_test[i])
        #print(error)
        err.append(error)
    mn=mean(err)
    print(mn)
    for i in range(0,len(y_pred)):
        mnsqerror=(y_pred[i]-mn)*(y_pred[i]-mn)
    x=mnsqerror/len(y_pred-1)
    return np.sqrt(x)
    
    
path='/scratch/Trainee_DATA/Juhi/'
aa=np.loadtxt(path+'database_new25.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

plot(y_pred,y_test)




values = pd.Series(y)
values.plot(kind='kde')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

def accuracy(y_pred,y_test):
    predRight = 0 #create count variables
    predWrong = 0
    for i in range(0,len(y_pred)):
        if y_pred[i] == y_test[i]:
            predRight += 1
        else:
            predWrong += 1
    print(predRight,predWrong)
    mr,mw = float(predRight), float(predWrong)
    acc = (mr/(mw+mr))*100 #calculate balanced accuracy, or average of sensitivty and specificity
    print('Accuracy='+ str(acc))

def mse(y_pred,y_test):
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

def plot(y_pred,y_test):
    plt.plot(y_test, y_pred, '.')#actual values of x and y
    plt.title('Elastic Net Regression')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()
    
aa=np.loadtxt('database.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

model=ElasticNet()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


accuracy(y_pred,y_test)
mse(y_pred,y_test)
plot(y_pred,y_test)

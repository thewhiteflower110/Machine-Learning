import numpy as np ##calculate mean and standard deviation
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt

aa=np.loadtxt('database.dat',unpack=True)
data=aa.T
data2=data[~np.isnan(data).any(axis=1)]
y=np.squeeze(data2[:,3])
x=np.squeeze(data2[:,4:])

j=1
lin=LinearRegression()
y=x[:,j]
x2=np.copy(x)
x2=np.delete(x2,j,axis=1)
lin.fit(x2,y)
y_pred=lin.predict(x2)
print('bt'+str(j)+':'+str(metrics.r2_score(y,y_pred)))

plt.plot(y, y_pred, '.')#actual values of x and y
plt.title('Linear Regression')
plt.xlabel('Y')
plt.ylabel('Y_pred')

plt.show()
#for j in range(0,24):
#    y=x[:,j]
#    x2=np.copy(x)
#    x2=np.delete(x2,j,axis=1)
#    lin.fit(x2,y)
#    y_pred=lin.predict(x2)
#    print('bt'+str(j)+':'+str(metrics.r2_score(y,y_pred)))
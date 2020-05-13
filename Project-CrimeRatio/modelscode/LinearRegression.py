import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics

path='/scratch/Trainee_DATA/Juhi/cyclone_files/'

aa1=np.loadtxt(path+'mix_testing_cyclone_set.txt',unpack=True) #contains the files to be taken in testing set
data1=aa1.T
x_train=data1[:,4:]
y_train=data1[:,3]

aa1=np.loadtxt(path+'mix_training_cyclone_set.txt',unpack=True) #contains the files to be taken in testing set
data1=aa1.T
x_test=data1[:,4:]
y_test=data1[:,3]


lin=LinearRegression()
lin.fit(x_train,y_train)
y_pred=lin.predict(x_test)
print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

plt.plot(y_test, y_pred, '.')#actual values of x and y
plt.title('Linear Regression')
plt.xlabel('Y_test')
plt.ylabel('Y_pred')
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:02:57 2020

@author: guest
"""
import matplotlib.pyplot as plt

mse=[18.3114579593,18.8312452248,18.4158044875,18.4284593436,18.439722059,17.433684558771304,14.943786696327996,35.8273092756,18.4397244858,16.5807560137,16.4432989691,17.5257731959,15.7216494845,17.4054982818,18.058419244,20.3951890034,16.3058419244]
acc=[0.0,0.0,0.0,0.0,0.0,0.0,0.343642611684,0.0,0.0,19.9312714777,21.3058419244,21.3058419244,19.587628866,17.1821305842,17.8694158076,16.1512027491,21.3058419244]
method=["ElasticNet Regression ","Lasso Regression-alpha_100","Lasso Regression-alpha_0.01","Ridge Regression-alpha_50 ","Ridge Regression-alpha_0.01 ","Boosting Regression Tree","Regression Tree","Linear Regression and Polynomial features","Linear Regression","Pca and Multinomial Regression using pipeline","Pca and Multinomial Regression-24","Pca and Multinomial Regression-20","Pca and Multinomial Regression-10","Pca and Multinomial Regression-3","Pca and Multinomial Regression-2","Pca and Multinomial Regression-1","multinomial Logistic regression"]
plt.plot(acc,c='b',linewidth=2,label="Accuracy")
plt.plot(mse,c='r',linewidth=2,label="MSE")
#plt.plot(min(mse))

plt.title('Acc and mse of different regression techniques')
plt.xlabel('Various methods')
plt.ylabel('scale for mse/acc')
plt.legend()
plt.show()
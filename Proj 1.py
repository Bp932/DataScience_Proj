# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:22:14 2022

@author: Bhavay Pant
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r"D:\IIT ISM Dhanbad\Machine Learning\ML Lab\DataBases\basic_database.csv")
print(df)

x = df[['X1']]
y= df[['y']]
plt.scatter(x,y)
plt.xlabel('area')
plt.ylabel('price')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
print("linear regression intercept: ",regressor.intercept_)
print("linear regression coefficient: ",regressor.coef_)
target_pred=regressor.predict(x)
print("predicted values: ",target_pred)

plt.plot(x,target_pred,color='g')
plt.scatter(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(x_test)
regressor.fit(x_train,y_train)
print(regressor.intercept_)

target_pred = regressor.predict(x_test)
print(target_pred)

from sklearn import metrics

print('Mean Absolute Error : ',metrics.mean_absolute_error(y_test, target_pred))
print('Mean Square Error : ',metrics.mean_squared_error(y_test,target_pred))
print('Root Mean Square Error : ',np.sqrt(metrics.mean_squared_error(y_test,target_pred)))

from sklearn.metrics import r2_score
r2_score(y_test,target_pred)


df = pd.read_csv(r"D:\IIT ISM Dhanbad\Machine Learning\ML Lab\DataBases\basic_database2.csv")
df
pd.plotting.scatter_matrix(df)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(df[['area','bedroom','age']],df.price)
regressor.intercept_
regressor.predict([[3000,3,40]])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['area','bedroom','age']],df.price,test_size=2,random_state=0)
regressor.fit(x_train,y_train)
regressor.intercept_
regressor.coef_

target_pred = regressor.predict(x_test)
print(target_pred)

print('Mean Absolute Error : ',metrics.mean_absolute_error(y_test, target_pred))
print('Mean Square Error : ',metrics.mean_squared_error(y_test,target_pred))
print('Root Mean Square Error : ',np.sqrt(metrics.mean_squared_error(y_test,target_pred)))


from sklearn.metrics import r2_score
r2_score(y_test,target_pred)

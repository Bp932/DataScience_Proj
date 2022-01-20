# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:53:41 2022

@author: Bhavay Pant
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r"D:\IIT ISM Dhanbad\Machine Learning\ML Lab\DataBases\insurance.csv")
df
df_dummy = pd.get_dummies(df.sex)
df_dummy = pd.concat([df,df_dummy],axis="columns")
df_dummy = df_dummy.drop(['sex','male'],axis='columns')

df_dummy2 = pd.get_dummies(df_dummy[['smoker','region']])
df_dummy = pd.concat([df_dummy,df_dummy2],axis="columns")
df_dummy = df_dummy.drop(['region_southwest','region','smoker_yes','smoker'],axis='columns')

from sklearn.linear_model import LinearRegression

x= df_dummy[['age','children','female','smoker_no','region_northeast','region_northwest','region_southeast']]
y= df_dummy.charges

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
regressor.intercept_
regressor.coef_

target_pred = regressor.predict(x_test)
target_pred

from sklearn import metrics

print('Mean Absolute Error : ',metrics.mean_absolute_error(y_test, target_pred))
print('Mean Square Error : ',metrics.mean_squared_error(y_test,target_pred))
print('Root Mean Square Error : ',np.sqrt(metrics.mean_squared_error(y_test,target_pred)))


from sklearn.metrics import r2_score
r2_score(y_test,target_pred)

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:34:25 2022

@author: Bhavay Pant
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer

df = pd.read_csv("D:\IIT ISM Dhanbad\Machine Learning\ML Lab\DataBases\Wine.csv")

df.style
df.dtypes
df.head()

df2 = df.drop('Customer_Segment',axis=1)
df2

from sklearn import datasets
wine = datasets.load_wine()
x = wine.data
y = wine.target
wine

pca = PCA(n_components=13)
decomp_reg = pca.fit(x).transform(x)

plt.scatter(decomp_reg[y == 0,3],decomp_reg[y == 0,0],s=80,c='orange',label='Type 0')
plt.scatter(decomp_reg[y == 1,3],decomp_reg[y == 1,0],s=80,c='yellow',label='Type 1')
plt.scatter(decomp_reg[y == 2,3],decomp_reg[y == 2,0],s=80,c='green',label='Type 2')
plt.title('PCA plot for Wine Dataset')


grr = pd.plotting.scatter_matrix(df,marker='o',hist_kwds={'bins': 20},figsize=(15, 15))

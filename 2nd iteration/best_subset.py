import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

import prediction as predict
import utils as util
import dataProcessing as dp

import time


'''' upload data '''
data_nov = pd.read_excel('../../Data/BLB_data_only_values_1511.xlsx')
data_dec = pd.read_excel('../../Data/BLB_data_only_values_1512.xlsx')
data_jan = pd.read_excel('../../Data/BLB_data_only_values_1501.xlsx')

''' extract features that express trend movement (substracting one month from another doesn't bring the whole column to zero)'''
numeric_nov = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]
numeric_dec = data_dec[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]
numeric_jan = data_jan[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]

numeric_nov_all = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'total assets', 'analyst rating']]
numeric_nov_all_no_anr = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'total assets']]
list = ['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'total assets']
results = []

numeric_nov_all = numeric_nov_all.dropna(axis=0, how='any')

y = numeric_nov_all.loc[:, 'analyst rating']
X = numeric_nov_all.drop(['analyst rating'], axis=1)

models = pd.DataFrame(columns=["model", "RSS", "R_sqr"])
tic = time.time()
for i in range(1,18):
    models.loc[i] = util.getBest(X, y, i)
toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(models["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('Residual Sum of Squares')
plt.subplot(2, 1, 2)
rsquared = models["R_sqr"]
plt.plot(rsquared)
plt.plot(rsquared.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('R squared')
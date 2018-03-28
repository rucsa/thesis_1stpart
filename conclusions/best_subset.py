import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import utils as util
import dataProcessing as dp

import time

X = pd.read_hdf('data.hdf5', 'Datataset1/X')

features = list(X.columns.values)
results = []

X = X.dropna(axis=0, how='any')

y = X.loc[:, 'analyst rating']
X = X.drop(['analyst rating'], axis=1)

models = pd.DataFrame(columns=["model", "RSS", "R_sqr"])
tic = time.time()
for i in range(1,20):
    models.loc[i] = util.getBest(X, y, i, LinearRegression())
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
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

import utils as util
from dataProcessing import nov_processed, y
from random import shuffle
import time

results = []
X = pd.read_hdf('dataWithOUTliers.hdf5', 'Datataset1/X')
features = list(X.columns.values)
y = X.loc[:,['analyst rating']].values.ravel()
X = X.drop(['analyst rating'], axis=1)

# shuffle columns of df
cols = X.columns.tolist()
shuffle(cols)
X = X[cols]
features = list(X.columns.values)
models_fwd = pd.DataFrame(columns=['model', 'RSS', 'R_squared', 'features'])
tic = time.time()
predictors = []
model = LinearRegression(fit_intercept=True, normalize=False, 
                                                    copy_X=True, n_jobs=1)
#model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1,  random_state=0, loss='ls')
##model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2, max_features='auto',  
#                              max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
#                              min_samples_leaf=1, min_samples_split=2,
#                              min_weight_fraction_leaf=0.0, 
#                              n_estimators=10, n_jobs=1,
#                              oob_score=False, random_state=0, 
#                              verbose=0, warm_start=False)
#model = Lasso(alpha=0.22, fit_intercept=True, normalize=False, precompute=False, 
#                              copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, 
#                              random_state=None, selection='cyclic')
#model = LassoLars(alpha=0.22, fit_intercept=True, verbose=False, normalize=True, 
#                  precompute='auto', max_iter=1000, eps=2.2204460492503131e-16, 
#                  copy_X=True, fit_path=True, positive=False)

for i in range(1, len(X.columns)+1):    #
    models_fwd.loc[i] = util.forward(X, y, predictors, model)
    predictors = models_fwd.loc[i]['features']

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

plt.figure(figsize=(8,4))
plt.subplot(2,1,1)
plt.plot(models_fwd["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('Residual Sum of Squares')
plt.subplot(2, 1, 2)
rsquared = models_fwd["R_squared"]
plt.plot(rsquared)
plt.plot(rsquared.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('R squared')
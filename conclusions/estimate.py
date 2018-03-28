from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLars, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


''' choose data set'''
# X0
X_train = pd.read_hdf('X_train.hdf5', 'Datataset1/X')
X_test = pd.read_hdf('X_test.hdf5', 'Datataset1/X')
y_train = pd.read_hdf('y_train.hdf5', 'Datataset1/X')
y_test = pd.read_hdf('y_test.hdf5', 'Datataset1/X')
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

alpha = 7.854037922617064e-05
# ([ 11.0619736 ,  12.13476388,  12.69785292,  12.0761462 ,
#        11.93870722,  14.57887634,  22.64929597,  12.79536465,
#        12.22676272,  10.09842659])
#mean mse = 0.38063117376012856

models = []
models.append(('LinearRegression', LinearRegression(fit_intercept=True, normalize=False, 
                                                    copy_X=True, n_jobs=1)))
models.append(('Lasso', Lasso(alpha=alpha, fit_intercept=True, normalize=False, precompute=False, 
                              copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, 
                              random_state=None, selection='cyclic')))

models.append(('GradientBoostingRegressor', GradientBoostingRegressor(n_estimators=100, 
                                                             learning_rate=0.1, 
                                                             max_depth=1, 
                                                             random_state=0, 
                                                             loss='ls')))
models.append(('RandomForestRegressor', RandomForestRegressor(bootstrap=True, 
                                                         criterion='mse', 
                                                         max_depth=2, 
                                                         max_features='auto', 
                                                         max_leaf_nodes=None,
                                                         min_impurity_decrease=0.0, 
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1, 
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0, 
                                                         n_estimators=10, n_jobs=1,
                                                         oob_score=False, random_state=0, 
                                                         verbose=0, warm_start=False)))


results = []
for name, model in models:
    prediction = model.fit(X_train, y_train).predict(X_test)
    
    y_mean = np.mean(y_test)
    mse = np.mean((y_test - prediction)**2)
    rss = mse * (y_test.size)
    
    ss_res = np.sum((y_test - prediction)**2)
    ss_tot = np.sum((y_test - y_mean)**2)
    
    r2 = 1-(ss_res/ss_tot)
    results.append({"model":name, 
#                    "RSS train":np.mean(RSS_train), 
                "RSS test":round(np.mean(rss),3), 
#                    "MSE train":MSE_train, 
                "MSE test":round(mse,3), 
#                    "R_squared train":r2_train, 
                "R_squared test":round(r2,3)})
    
avg_pred = np.zeros(y_test.size)
average_pred = np.average(y_test)
for i in range (0, avg_pred.size):
    avg_pred[i] = average_pred
    
residuals = y_test - avg_pred
mean_obs = np.mean(y_test)
ss_res = np.sum((y_test - avg_pred)**2)
ss_tot = np.sum((y_test - mean_obs)**2)
mse = np.mean((avg_pred - y_test)**2)
rss = mse * (y_test.size)
r2 = 1-(ss_res/ss_tot)


results.append({"model":"Average (Benchmark)", 
                    #"RSS train":rss, 
                    "RSS test":round(rss,3), 
                    #"MSE train":mse, 
                    "MSE test":round(mse,3), 
                    #"R_squared train":r2, 
                    "R_squared test":round(r2,3)})    
    
results = pd.DataFrame(results)
results.to_excel('estimates.xlsx', sheet_name='Sheet2')
results.to_hdf('estimates.hdf5', 'Datataset2/X')
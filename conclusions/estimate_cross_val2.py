from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLars, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasso_cross_val import alpha



''' set data set in lasso_cross_val'''
#
#X_train = pd.read_hdf('X_train.hdf5', 'Datataset1/X')
#X_test = pd.read_hdf('X_test.hdf5', 'Datataset1/X')
#y_train = pd.read_hdf('y_train.hdf5', 'Datataset1/X')
#y_test = pd.read_hdf('y_test.hdf5', 'Datataset1/X')
#y_train = y_train.values.flatten()
#y_test = y_test.values.flatten()
## 0.37683377357012426


X = pd.read_hdf('X.hdf5', 'Datataset1/X')

y = X[['analyst rating']].values.flatten()
X = X.drop(['analyst rating'], axis=1)

models = []
models.append(('LinearRegression', LinearRegression(fit_intercept=True, normalize=False, 
                                                    copy_X=True, n_jobs=1)))
models.append(('Lasso', Lasso(alpha=alpha, fit_intercept=True, normalize=False, precompute=False, 
                              copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, 
                              random_state=None, selection='cyclic')))

models.append(('GradientBoostingRegressor',  GradientBoostingRegressor(loss='ls', 
                                 learning_rate=0.06, 
                                 n_estimators=100, 
                                 subsample=0.6, 
                                 criterion='friedman_mse', 
                                 min_samples_split=3,
                                 min_samples_leaf=3, 
                                 min_weight_fraction_leaf=0.0, 
                                 max_depth=3, 
                                 min_impurity_decrease=0.0, 
                                 min_impurity_split=None, 
                                 init=None, 
                                 random_state=None,
                                 max_features=None, 
                                 alpha=0.04, 
                                 verbose=0, 
                                 max_leaf_nodes=None, 
                                 warm_start=False, 
                                 presort='auto')))
models.append(('RandomForestRegressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5, 
                              max_features='auto', 
                              max_leaf_nodes=None, 
                              min_impurity_decrease=0.0, 
                              min_impurity_split=None, 
                              min_samples_leaf=50, 
                              min_samples_split=2, 
                              min_weight_fraction_leaf=0.0, 
                              n_estimators=100, 
                              n_jobs=-1, 
                              oob_score=True, 
                              random_state=0, 
                              verbose=0, 
                              warm_start=False)))
models.append(('PolynomialFeatures', Pipeline([('poly', PolynomialFeatures(degree=1, interaction_only=True, include_bias=True)),
                                               ('linear', Lasso(alpha=alpha, fit_intercept=True, normalize=False, precompute=False, 
                                                                copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, 
                                                                random_state=None, selection='cyclic'))])))


results = []
for name, model in models:
    scores = cross_validate(model, X, y, cv=10, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
    
    RSS_train = -len(y)*scores['train_neg_mean_squared_error']
    MSE_train = -np.mean(scores['train_neg_mean_squared_error'])
    r2_train = np.mean(scores['train_r2'])
    
    RSS_test = -len(y)*scores['test_neg_mean_squared_error']
    MSE_test = -np.mean(scores['test_neg_mean_squared_error'])
    r2_test = np.mean(scores['test_r2'])
    
    results.append({"model":name, 
                    "RSS test":round(np.mean(RSS_test), 3), 
                    "MSE test":round(MSE_test, 3), 
                    "R_squared test":round(r2_test,3)})
    
    
avg_pred = np.zeros(y.size)
average_pred = np.average(y)
for i in range (0, avg_pred.size):
    avg_pred[i] = average_pred
    
residuals = y - avg_pred
mean_obs = np.mean(y)
ss_res = np.sum((y - avg_pred)**2)
ss_tot = np.sum((y - mean_obs)**2)
mse = np.mean((avg_pred - y)**2)
rss = mse * (y.size)
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
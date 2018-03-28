from dataProcessing import y
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_hdf('data.hdf5', 'Datataset1/X')
y = pd.DataFrame(y).values.ravel()


full_feature_list = ['revenue', 'return last 3 month', 'volatility 30 days',
                     'gross profit', 'quick ratio', 'P/E', 'returns last 6 months',
                     'volatility 360 days', 'return last year', 'market cap',
                     'net income', 'EPS', 'operational cash flow', 'ethics',
                     'bribery', 'Sector', 'returns last 5 years', 'region',
                     'PSR', 'size', 'total assets', 'inventory turnover',
                     'adjusted beta', 'volatility 90 days']

# eliminate features
delete_list = ['revenue', 'return last 3 month',
                     'quick ratio', 'P/E',
                     'volatility 360 days', 'return last year', 'market cap',
                     'EPS', 'ethics',
                     'bribery', 'Sector', 'region',
                     'PSR', 'size', 'total assets', 'inventory turnover',
                     'adjusted beta', 'volatility 90 days']
delete_list = ['P/E', 'market cap', 'inventory turnover', 'revenue', 'size', 
               'Sector', 'region', 'ethics', 'bribery', 'PSR'] # data analysis
delete_list = ['adjusted beta', 'volatility 90 days', 'return last 3 month', 
               'EPS', 'quick ratio', 'inventory turnover', 'revenue', 
               'operational cash flow', 'total assets', 'Sector', 'region', 
               'ethics', 'bribery', 'size'] #random forest feature importance selection
delete_list = ['adjusted beta', 'volatility 90 days', 'market cap', 'net income',
               'EPS', 'quick ratio', 'revenue', 'returns last 6 months', 'volatility 360 days',
               'P/E', 'volatility 30 days', 'operational cash flow', 
               'total assets', 'Sector', 'bribery'] # linear regression feature correlation
delete_list = ['volatility 30 days', 'volatility 360 days', 'P/E', 'returns last 5 years', 
               'Sector', 'region', 'PSR'] # lasso feature selection





X = X.drop(delete_list, axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state=0)

models = []
models.append(('LinearRegression', LinearRegression(fit_intercept=True, normalize=False, 
                                                    copy_X=True, n_jobs=1)))
models.append(('LassoLars', LassoLars(alpha=0.073, fit_intercept=True, verbose=False, 
                                      normalize=True, precompute=False, max_iter=1000, 
                                      eps=2.2204460492503131e-16, copy_X=True, fit_path=True, 
                                      positive=False)))
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
    scores = cross_validate(model, X, y, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
    
    RSS_train = -len(y)*scores['train_neg_mean_squared_error']
    MSE_train = -np.mean(scores['train_neg_mean_squared_error'])
    r2_train = np.mean(scores['train_r2'])
    
    RSS_test = -len(y)*scores['test_neg_mean_squared_error']
    MSE_test = -np.mean(scores['test_neg_mean_squared_error'])
    r2_test = np.mean(scores['test_r2'])
    
    results.append({"model":name, 
                    "RSS train":np.mean(RSS_train), "RSS test":np.mean(RSS_test), 
                    "MSE train":MSE_train, "MSE test":MSE_test, 
                    "R_squared train":r2_train, "R_squared test":r2_test})
results = pd.DataFrame(results)
results.to_excel('estimates.xlsx', sheet_name='Sheet2')
results.to_hdf('estimates.hdf5', 'Datataset2/X')
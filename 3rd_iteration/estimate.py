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
y = pd.DataFrame(y)



full_feature_list = ['revenu', 'return last 3 month', 'volatility 30 days',
                     'gross profit', 'quick ratio', 'P/E', 'returns last 6 months',
                     'volatility 360 days', 'return last year', 'market cap',
                     'net income', 'EPS', 'operational cash flow', 'ethics',
                     'bribery', 'returns last 5 years', 
                     'PSR', 'size', 'total assets', 'inventory turnover',
                     'adjusted beta', 'volatility 90 days']

# eliminate features
delete_list = ['revenu', 'return last 3 month',
                     'quick ratio', 'P/E',
                     'volatility 360 days', 'return last year', 'market cap',
                     'EPS', 'ethics',
                     'bribery', 
                     'PSR', 'size', 'total assets', 'inventory turnover',
                     'adjusted beta', 'volatility 90 days']
delete_list = ['P/E', 'market cap', 'inventory turnover', 'revenu', 'size', 
              'ethics', 'bribery', 'PSR'] # data analysis
delete_list = ['adjusted beta', 'volatility 90 days', 'return last 3 month', 
               'EPS', 'quick ratio', 'inventory turnover', 'revenu', 
               'operational cash flow', 'total assets',
               'ethics', 'bribery', 'size'] #random forest feature importance selection
delete_list = ['adjusted beta', 'volatility 90 days', 'market cap', 'net income',
               'EPS', 'quick ratio', 'revenu', 'returns last 6 months', 'volatility 360 days',
               'P/E', 'volatility 30 days', 'operational cash flow', 
               'total assets', 'bribery'] # linear regression feature correlation
delete_list = ['volatility 30 days', 'volatility 360 days', 'P/E', 'returns last 5 years', 
              'PSR'] # lasso feature selection


X = X.loc[:, ['return last year', 'adjusted beta', 'market cap', 'analyst rating']]
#X = pd.concat([X, y], axis=1)
X = X.dropna(axis=0, how='any')
X = X.loc[X['analyst rating'] > 0]

#X = X.loc[X['return last year'] <= 30]
#X = X.loc[X['return last year'] >= -20]
#X = X.loc[X['adjusted beta'] <= 2]
#X = X.loc[X['adjusted beta'] >= -2]
#X = X.loc[X['market cap'] <= 14000]
#X = X.loc[X['market cap'] >= 1000]

X = X.sort_values(by='analyst rating', axis=0).reset_index(drop=True)

''' over and undersampling'''
if False:
    # balance strong sells
    for i in range (1, 200):
        X = X.append(X[0:2])
        
    #balance sells
    for i in range (1, 12):
        X = X.append(X[2:37])
    #X = X.drop(delete_list, axis=1)
    
    #balance buy
    count = 0
    for index, row in X.iterrows():
        if ((row['analyst rating']>=3.5) & (row['analyst rating']<4.5)):
            if(count % 2 == 0):
                X = X.drop(index)
            count=count+1
            
    #balance strong buy
    for index, row in X.iterrows():
        if ((row['analyst rating']>=4.5) & (row['analyst rating']<=5)):   
            X = X.append(row)
    
    #[0, 1.5, 2.5, 3.5, 4.5, 5]
    y = X.loc[:, ['analyst rating']]
    y = y.values.ravel()
    a, b, c, d, e = 0, 0, 0, 0, 0
    for i in y:
        i = float(i)
        if (i>=0) & (i<1.5): 
            a=a+1
        elif (i>=1.5) & (i<2.5): 
            b=b+1
        elif (i>=2.5) & (i<3.5): 
            c=c+1
        elif (i>=3.5) & (i<4.5):
            d=d+1
        else: 
            e=e+1
    print ("target contains /n {} strong sell /n {} sell /n {} hold /n {} buy /n {} strong buy".format(a, b, c, d, e))




X.to_hdf('balanced.hdf5', 'Datataset3/X')
y = X.loc[:, ['analyst rating']]
y = y.values.ravel()
X = X.drop(['analyst rating'], axis=1)
X = X.sample(frac=1).reset_index(drop=True)

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
                    "RSS train":rss, "RSS test":rss, 
                    "MSE train":mse, "MSE test":mse, 
                    "R_squared train":r2, "R_squared test":r2})    
    
results = pd.DataFrame(results)
results.to_excel('estimates.xlsx', sheet_name='Sheet2')
results.to_hdf('estimates.hdf5', 'Datataset2/X')
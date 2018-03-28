from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLars, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasso_cross_val import alpha

X_train = pd.read_hdf('X_train.hdf5', 'Datataset1/X')
X_test = pd.read_hdf('X_test.hdf5', 'Datataset1/X')
y_train = pd.read_hdf('y_train.hdf5', 'Datataset1/X')
y_test = pd.read_hdf('y_test.hdf5', 'Datataset1/X')
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

if True:  
    a, b, c, d, e = 0, 0, 0, 0, 0
    for i in y_train:
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
    print ("target contains {}-strong sell, {}-sell, {}-hold, {}-buy and {}-strong buy".format(a, b, c, d, e))
    
''' over and undersampling'''
if True:
    #order samples
    dfy = pd.DataFrame(y_train, columns=['analyst rating'])
    dfy.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_train = pd.concat([X_train, dfy], axis = 1)
    X_train = X_train.sort_values(by='analyst rating', axis=0).reset_index(drop=True)
    
    # balance strong sells
    for i in range (1, 50):
        X_train = X_train.append(X_train[0:1])
        
    #balance sells
    for i in range (1, 2):
        X_train = X_train.append(X_train[1:24])
    X_train = X_train.append(X_train[11:12])
    X_train = X_train.append(X_train[14:15])
    X_train = X_train.append(X_train[21:22])
    X_train = X_train.append(X_train[22:23])
    
    #balance hold
    count = 0
    for index, row in X_train.iterrows():
        if ((row['analyst rating']>=2.5) & (row['analyst rating']<3.5)):
            if(count % 5 != 0):
                X_train = X_train.drop(index)
            count=count+1
    
    #balance buy
    count = 0
    for index, row in X_train.iterrows():
        if ((row['analyst rating']>=3.5) & (row['analyst rating']<4.5)):
            if(count % 10 != 0):
                X_train = X_train.drop(index)
            count=count+1
            
    #balance strong buy
    count = 0
    for index, row in X_train.iterrows():
        if ((row['analyst rating']>=4.5) & (row['analyst rating']<=5)):   
            if(count % 2 != 0 & count < 101):
                X_train = X_train.drop(index)
            elif(count >= 101):
                X_train = X_train.drop(index)
            count=count+1
            
    # shuffle back       
    y_train = X_train[['analyst rating']].values.ravel()
    X_train = X_train.drop(['analyst rating'], axis=1)
    X_train = X_train.sample(frac=1).reset_index(drop=True)
    
    
models = []
results = []
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

for name, model in models:
    prediction = model.fit(X_train, y_train).predict(X_test)
    
    y_mean = np.mean(y_test)
    mse = np.mean((y_test - prediction)**2)
    rss = mse * (y_test.size)
    
    ss_res = np.sum((y_test - prediction)**2)
    ss_tot = np.sum((y_test - y_mean)**2)
    
    r2 = 1-(ss_res/ss_tot)
    
    results.append({"model":name,  
                        "RSS":round(np.mean(rss),3), 
                        "MSE":round(mse,3),  
                        "R_squared":round(r2,3)})
    
avg_pred = np.zeros(y_test.size)
average_pred = np.average(y_train)
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
                    "RSS":round(rss,3), 
                    "MSE":round(mse,3), 
                    "R_squared":round(r2,3)})   
results = pd.DataFrame(results)
results.to_excel('estimates.xlsx', sheet_name='Sheet2')

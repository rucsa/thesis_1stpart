import dataProcessing as dp
import pandas as pd
import numpy as np
from sklearn import linear_model, svm, neighbors
from sklearn. linear_model import SGDRegressor
from sklearn import preprocessing, metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import matplotlib
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import iqr
import math
from scipy import stats
from sklearn.linear_model import LinearRegression
import collections
import prediction as predict
import xlsxwriter


'''' upload data '''
data_nov = pd.read_excel('../../Data/BLB_data_only_values_1511.xlsx')
data_dec = pd.read_excel('../../Data/BLB_data_only_values_1512.xlsx')
data_jan = pd.read_excel('../../Data/BLB_data_only_values_1501.xlsx')

''' extract features that express trend movement (substracting one month from another doesn't bring the whole column to zero)'''
numeric_nov = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]
numeric_dec = data_dec[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]
numeric_jan = data_jan[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]

numeric_nov_all = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'revenue', 'gross profit', 'net income', 'operational cash flow', 'total assets', 'analyst rating']]
numeric_nov_all_no_anr = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'revenue', 'gross profit', 'net income', 'operational cash flow', 'total assets']]
list = ['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'revenue', 'gross profit', 'net income', 'operational cash flow', 'total assets']

results = {}

writer = pd.ExcelWriter('../results/results.xlsx', engine='xlsxwriter')

''' plot linear regression trendline '''
''' individual feature correlation - without outliers'''
output = pd.DataFrame(columns=['Feature', 'R_squared', 'Correlation'])
index = 0
for item in list:
    p = numeric_nov_all_no_anr[[item]].dropna().values
    #min = np.percentile(p, 25)
    max = np.percentile(p, 75)
    #print '%s values should be between %s and %s' % (item, 0, max)
    pe = data_nov[[item, 'analyst rating']]
    pe = pe[np.isfinite(pe[item])]
    pe = pe.loc[pe[item] <= max]
    anr = pe[['analyst rating']].values
    pe = pe[[item]].values
    #fig = plt.figure()
    #plt.ylabel('Analyst rating')
    #plt.xlabel(item)
    # linregress needs one dimentional array - obtained through slicing
    #plt.scatter(pe, anr)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(pe[:,0], anr[:,0])
    #plt.plot(pe, intercept + slope*pe, 'r', label='fitted line')
    #plt.show()
    nl = '\n'
    output.loc[index] = ({"Feature": item, "R_squared": rvalue**2, "Correlation": rvalue})
    index = index + 1

output.to_excel(writer, sheet_name='Sheet1')

''' individual feature correlation - with outliers'''
numeric_nov_all.corr()['analyst rating'].to_excel(writer, sheet_name='Sheet2')

writer.save()
writer.close()

output_dec = data_dec[['analyst rating']]
output_jan = data_jan[['analyst rating']]

# deal with missing values
numeric_nov = dp.interpolate(numeric_nov)
numeric_dec = dp.interpolate(numeric_dec)
numeric_jan = dp.interpolate(numeric_jan)

# extract trends
dec_nov = numeric_dec.sub(numeric_nov, axis=0)
jan_dec = numeric_jan.sub(numeric_dec, axis=0)

# add december data to trends
non_trends_dec = data_dec[[	'returns last 5 years',	'quick ratio',	'inventory turnover', 'revenue', 'gross profit', 'net income', 'operational cash flow', 'total assets', 'market cap']]
non_trends_dec = dp.interpolate(non_trends_dec)

# make fundamental data relative to company size
non_trends_dec.loc[:, 'total assets'] = non_trends_dec.loc[:, 'total assets'] / 1000 # express in billions
non_trends_dec.loc[:, 'inventory turnover'] = non_trends_dec.loc[:, 'inventory turnover'] / non_trends_dec.loc[:, 'market cap']
non_trends_dec.loc[:, 'revenue'] = non_trends_dec.loc[:, 'revenue'] / non_trends_dec.loc[:, 'market cap']
non_trends_dec.loc[:, 'gross profit'] = non_trends_dec.loc[:,'gross profit'] / non_trends_dec.loc[:,'market cap']
non_trends_dec.loc[:, 'net income'] = non_trends_dec.loc[:,'net income'] / non_trends_dec.loc[:,'market cap']
non_trends_dec.loc[:, 'operational cash flow'] = non_trends_dec.loc[:,'operational cash flow'] / non_trends_dec.loc[:,'market cap']
non_trends_dec = non_trends_dec.drop('market cap', 1)

# concat data frames
dec_nov_all = pd.concat([dec_nov, non_trends_dec], axis=1)

# make arrays
dec_nov = dec_nov.values
jan_dec = jan_dec.values
output_dec = output_dec.values.ravel()
output_jan = output_jan.values.ravel()

# standardization
dec_nov_stand = preprocessing.scale(dec_nov)

# scale [0, 1]
minmax_scaler = preprocessing.MinMaxScaler()
dec_nov_scal = minmax_scaler.fit_transform(dec_nov)

# scale [-1, 1]
max_abs_scaler = preprocessing.MaxAbsScaler()
dec_nov_abs_scal = max_abs_scaler.fit_transform(dec_nov_stand)

# non-linear transformation
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
dec_nov_non_lin = quantile_transformer.fit_transform(dec_nov)

# normalization
dec_nov_norm = preprocessing.normalize(dec_nov, norm='l2')

models = []
models.append(('REG', linear_model.LinearRegression()))
models.append(('LSS', linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')))
models.append(('LSVR', svm.LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='squared_epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)))
models.append(('SVR', svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)))
models.append(('NuSVR', svm.NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)))
models.append(('KNN', neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30)))
models.append(('TREE', DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)))
models.append(('GTB', GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')))
models.append(('MPR', MLPRegressor(hidden_layer_sizes=(5, 4), activation='tanh', solver='lbfgs', alpha=0.4, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=False, random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)))

# predict.regression(models, dec_nov_scal, output_dec)

# TODO: Validation curves
# TODO: Learning curves

# http://sdsawtelle.github.io/blog/output/week6-andrew-ng-machine-learning-with-python.html



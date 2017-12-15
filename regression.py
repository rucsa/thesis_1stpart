import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, linear_model, metrics, svm, neighbors, cross_validation
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

##### files for output
f = open('output.txt', 'w').close()
f = open('output.txt', 'a')

##### upload data
data = pd.read_csv('../Data/model_data.csv', sep=';', header = 0);

##### clean
data = data.replace({'%':''}, regex=True);

##### replace NaN values with 0
data = data.fillna(0)

##### split data into data frames
##### 'gross profit', 'net income', 'operational cash flow', 'market cap', 'total assets' expressed in millions
fund_data = data[['quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'P/E', 'EPS', 'market cap', 'total assets', 'number of employees']]
fund_string_data = data [['Ticker symbol', 'Security', 'Sector', 'region', 'ethics', 'bribery']]
mark_data = data[['raw beta on SPX market', 'adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'returns last 5 years']]
output = data[['analyst rating']]
all_int_data = data[['quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'P/E', 'EPS', 'market cap', 'total assets', 'number of employees', 'raw beta on SPX market', 'adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'returns last 5 years', 'analyst rating']]

## mixed features for second experiment
# eps redundant information
set1 = data [['operational cash flow', 'P/E', 'EPS', 'adjusted beta', 'return last 3 month']]
set1.loc[:, 'operational cash flow'] = set1.loc[:,'operational cash flow'] / fund_data.loc[:,'market cap']
set2 = data [['volatility 360 days', 'adjusted beta', 'returns last 6 months']]
set3 = data [['P/E', 'adjusted beta', 'returns last 6 months', 'volatility 360 days' ]]
set4 = data [['EPS', 'adjusted beta', 'returns last 6 months', 'volatility 360 days' ]]

# make arrays
set1 = set1.values
set2 = set2.values
set3 = set3.values
set4 = set4.values
output = output.values.ravel()
#output_en = np.asarray(output['analyst rating'].values, dtype="|S6")

## TODO make separate script for classification with labelled Y
##### create bins for output variable
#bins = [0, 1, 2, 3, 4]
#labels = [strong sell, sell, hold, buy, strong buy]
# encode y for some classifiers
#en = preprocessing.LabelEncoder()
#en.fit(output.values)
#output_en = output['analyst rating'].values.ravel()

##### visualize
## TODO random forrest
## TODO move visualization to another file
## TODO improve visualization: boxplots & their histograms, define outliers, add colors
# styling of graphs
#colMap={0:"red", 1:"blue", 2:"yellow", 3:"green", 4:"blue", 5:"white", 6:"black", 7:"brown", 8:"pink"}
#plt.style.use('ggplot')

# histograms
#fund_data.hist()
#plt.show()
## DATA NOTES: data above is very skewed to the left

#mark_data.hist()
#plt.show()
## DATA NOTES: data fairly even distributed, exceptions skewed to the left: volatility 30 days, 90 days, 360 days, return 5 years, returns last year

pp = sns.pairplot(all_int_data, x_vars = ['quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'P/E', 'EPS', 'market cap', 'total assets', 'number of employees', 'raw beta on SPX market', 'adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'returns last 5 years'], y_vars = ["analyst rating"])
plt.show()

##### preprocessing

### make market data relative to company size
fund_data.loc[:, 'market cap'] = fund_data.loc[:, 'market cap'] / 1000 # express in billions
fund_data.loc[:, 'total assets'] = fund_data.loc[:, 'total assets'] / 1000 # express in billions
fund_data.loc[:, 'inventory turnover'] = fund_data.loc[:, 'inventory turnover'] / fund_data.loc[:, 'market cap']
fund_data.loc[:, 'sale ravenue turnover'] = fund_data.loc[:, 'sale ravenue turnover'] / fund_data.loc[:, 'market cap']
fund_data.loc[:, 'gross profit'] = fund_data.loc[:,'gross profit'] / fund_data.loc[:,'market cap']
fund_data.loc[:, 'net income'] = fund_data.loc[:,'net income'] / fund_data.loc[:,'market cap']
fund_data.loc[:, 'operational cash flow'] = fund_data.loc[:,'operational cash flow'] / fund_data.loc[:,'market cap']

##### check data
#f.write('\n fundamental analysis data..... type: ' + str(type(fund_data)) + '\n')
#fund_data.to_csv('output.txt', mode='a')
#print(fund_data.head())
#print('\n market data...type: ' + str(type(mark_data)) + '\n')
#print(mark_data.head())
#print('\n output data....type: ' + str(type(output)) + '\n')
#print(output.head())

# scatter matrix ## correlation with buy/hold/sell advice on y axis
#pd.plotting.scatter_matrix(mark_data, alpha=0.7, diagonal = 'hist', figsize=[8,8], s=100)
#plt.show()
#pd.plotting.scatter_matrix(mark_data, alpha=0.7, diagonal = 'hist', figsize=[8,8], s=100)
#plt.show()

### scale
scaler = preprocessing.StandardScaler()
scaled_mark = scaler.fit(mark_data).transform(mark_data)
scaled_fund = scaler.fit(fund_data).transform(fund_data)
#print('\n Market data scaled......')
#print(scaled_fund)

### scale to range [0, 1]
minmax_scaler = preprocessing.MinMaxScaler()
range_mark = minmax_scaler.fit_transform(mark_data)
range_fund = minmax_scaler.fit_transform(fund_data)
#print('\n Market data scaled to range [0,1]......')
#print(range_mark)

### scale to range [-1, 1] - good for data that is already centered to zero or it is sparse
max_abs_scaler = preprocessing.MaxAbsScaler()
range_abs_mark = max_abs_scaler.fit_transform(mark_data)
range_abs_fund = max_abs_scaler.fit_transform(fund_data)
#print('\n Market data scaled to range [-1,1]......')
#print(scaled_mark)

### non-linear transformation?
### normalization - useful if you plan to use a quadratic form such as dot product
#   l1 norm = minimizing the sum of absolute differences
#   l2 norm = minimizing the sum of the square of differences - inefficient on non sparse methods
norm_mark = preprocessing.normalize(mark_data, norm='l2')
norm_fund = preprocessing.normalize(fund_data, norm='l2')
#print('\n Market data normalized......')
#print(norm_mark)

### encoding
### generate polynomial features

##### split into training and test sets -  uses StandardScaler
#X_train, X_test, y_train, y_tets = train_test_split (scaled_fund, output, test_size =0.3, train_size=0.7, random_state=47, shuffle=True)
#output = output.values.ravel()

##### classify
## made with 5 folds, with 10 results are better
# ordinary least squares ___Mean squared error = 0.28, R^2 = 0.01
reg = linear_model.LinearRegression()

# logistic regression ______Mean squared error = 0.31, R^2 = -0.07
l_reg = linear_model.LogisticRegressionCV()

# lasso ____________________Mean squared error = 0.28, R^2 = 0.01
lss = linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

# linear SVR _______________Mean squared error = 0.38, R^2 = -0.34
lsvr = svm.LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='squared_epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)

# SVR ______________________Mean squared error = 0.31, R^2 = -0.07
svr = svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

# NuSVR ____________________Mean squared error = 0.31, R^2 = -0.07
nusvr = svm.NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)

# knn ______________________Mean squared error = 0.32, R^2 = -0.12
knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30)

# decision tree ____________Mean squared error = 0.52, R^2 = -0.83
tree = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# gradient tree boosting ___Mean squared error = 0.31, R^2 = -0.08
gtb = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

# multi-layer perceptron regressor
mpr = MLPRegressor(hidden_layer_sizes=(5, 4), activation='tanh', solver='lbfgs', alpha=0.4, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=False, random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# cross validation
# takes care of splitting data
predicted = cross_validation.cross_val_predict(mpr, set4, output, cv=10)
print("Mean squared error: %0.2f" % (metrics.mean_squared_error(output, predicted)))
print("Mean absolute error: %0.2f" % (metrics.mean_absolute_error(output, predicted)))
print("R^2 coefficient: %0.2f" % (metrics.r2_score(output, predicted)))


fig, ax = plt.subplots()
ax.scatter(output, predicted, edgecolors=(0, 0, 0))
ax.plot([output.min(), output.max()], [output.min(), output.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# TODO split, train, predict, validate
#print("Accuracy: " + metrics.accuracy_score(output, predicted))
#print("Classification report \n")
#print metrics.classification_report(output, predicted)
#clf = MLPClassifier (solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(2,2), random_state=1)
#clf.fit(mark_data, y)
#out = clf.predict ([[1, 10, 63915, 55689, 24733, 20196, 31, 2],[0,	2, 59387, 50209, 5400, 8222, 21, 6]] )
#print(out)


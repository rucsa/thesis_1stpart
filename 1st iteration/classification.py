import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect
from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing, linear_model, metrics, svm, neighbors, cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier


f = open('classes.txt', 'w').close()
f = open('classes.txt', 'a')

##### upload data
data = pd.read_csv('../Data/model_data.csv', sep=';', header = 0);

##### clean
data = data.replace({'%':''}, regex=True);

##### replace NaN values with 0
data = data.fillna(0)

##### split data into data frames
##### 'gross profit', 'net income', 'operational cash flow', 'market cap', 'total assets' expressed in millions
fundamentalData = data[['quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'P/E', 'EPS', 'market cap', 'total assets', 'number of employees']]
stringData = data [['Ticker symbol', 'Security', 'Sector', 'region', 'ethics', 'bribery']]
marketData = data[['raw beta on SPX market', 'adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'returns last 5 years']]
output = data[['analyst rating']]
allData = data[['quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'P/E', 'EPS', 'market cap', 'total assets', 'number of employees', 'raw beta on SPX market', 'adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'returns last 5 years', 'analyst rating']]

## mixed features for second experiment
set1 = data [['operational cash flow', 'P/E', 'EPS', 'adjusted beta', 'return last 3 month']]
set1.loc[:, 'operational cash flow'] = set1.loc[:,'operational cash flow'] / fundamentalData.loc[:,'market cap']
set2 = data [['volatility 360 days', 'adjusted beta', 'returns last 6 months']]
set3 = data [['P/E', 'adjusted beta', 'returns last 6 months', 'volatility 360 days' ]]
set4 = data [['EPS', 'adjusted beta', 'returns last 6 months', 'volatility 360 days' ]]

# make arrays
fundamentalData = fundamentalData.values
marketData = marketData.values
allData = allData.values
set1 = set1.values
set2 = set2.values
set3 = set3.values
set4 = set4.values
output = output.values.ravel()

# split data into bins
### with bisect
labels = ['strong sell', 'sell', 'hold', 'buy', 'strong buy', 'very strong buy']
breakpoints = [1, 2, 3, 4, 5]
def label(total):
	return labels[bisect(breakpoints, total)]
outputLabels = map(label, output)
f.write ('LEGEND: \n 0-1:A (strong sell) \n 1-2:B (sell) \n 2-3:C (hold) \n 3-4:D (buy)\n 4-5:E (strong buy) \n 5:F (very strong buy) \n \n')

### with pandas cut -- splitting is relative to data values - does not seem suited
#outputLabels = pd.cut(output, 5, right=True, labels=['strong sell', 'sell', 'hold', 'buy', 'strong buy' ], retbins=True, precision=3, include_lowest=True)
#outputLabels = np.asarray(outputLabels)

# validate bins labels in 'classes.txt' file
for i in range(1,len(output)):
	f.write(str(output[i]) + '      ' + str(outputLabels[i]) + '\n')

#### classify
# ordinary least squares
ols = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

# linear SVC ___________________________________________________set1___Accuracy = 0.50, Precision = 0.26, Recall = 0.26
#_______________________________________________________________set2___Accuracy = 0.50, Precision = 0.37, Recall = 0.30
#_______________________________________________________________set3___Accuracy = 0.48, Precision = 0.26, Recall = 0.30
#_______________________________________________________________set4___Accuracy = 0.47, Precision = 0.27, Recall = 0.29
#____________________________________________________fundamentalData___Accuracy = 0.31, Precision = 0.23, Recall = 0.24
#_________________________________________________________marketData___Accuracy = 0.46, Precision = 0.24, Recall = 0.25
lsvr = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

# Stochastic Gradient Descent with one-vs-all multiclassification _1___Accuracy = 0.49, Precision = 0.27, Recall = 0.27
#_______________________________________________________________set2___Accuracy = 0.45, Precision = 0.23, Recall = 0.24
#_______________________________________________________________set3___Accuracy = 0.43, Precision = 0.24, Recall = 0.24
#_______________________________________________________________set4___Accuracy = 0.45, Precision = 0.23, Recall = 0.24
#____________________________________________________fundamentalData___Accuracy = 0.49, Precision = 0.26, Recall = 0.26
#_________________________________________________________marketData___Accuracy = 0.41, Precision = 0.25, Recall = 0.25
sgdc = linear_model.SGDClassifier(loss='perceptron', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, n_iter=None)

# Multi class decision tree  ___________________________________set1___Accuracy = 0.47, Precision = 0.27, Recall = 0.27
#_______________________________________________________________set2___Accuracy = 0.45, Precision = 0.25, Recall = 0.25
#_______________________________________________________________set3___Accuracy = 0.47, Precision = 0.28, Recall = 0.29
#_______________________________________________________________set4___Accuracy = 0.44, Precision = 0.24, Recall = 0.24
#____________________________________________________fundamentalData___Accuracy = 0.46, Precision = 0.26, Recall = 0.26
#_________________________________________________________marketData___Accuracy = 0.48, Precision = 0.26, Recall = 0.28
dtc = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

# gaussian process classifier - multiclass one_vs_rest _________set1___Accuracy = 0.47, Precision = 0.27, Recall = 0.30
#_______________________________________________________________set2___Accuracy = 0.48, Precision = 0.27, Recall = 0.30
#_______________________________________________________________set3___Accuracy = 0.40, Precision = 0.25, Recall = 0.28
#_______________________________________________________________set4___Accuracy = 0.43, Precision = 0.26, Recall = 0.29
#____________________________________________________fundamentalData___Accuracy = 0.01, Precision = 0.00, Recall = 0.25
#_________________________________________________________marketData___Accuracy = 0.16, Precision = 0.33, Recall = 0.35
gpc = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=1)

# multinomial native bayes - input has negative values
mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

# gradient boosting classifier _________________________________set1___Accuracy = 0.49, Precision = 0.25, Recall = 0.26
#_______________________________________________________________set2___Accuracy = 0.49, Precision = 0.25, Recall = 0.26
#_______________________________________________________________set3___Accuracy = 0.53, Precision = 0.27, Recall = 0.28
#_______________________________________________________________set4___Accuracy = 0.52, Precision = 0.26, Recall = 0.28
#____________________________________________________fundamentalData___Accuracy = 0.57, Precision = 0.33, Recall = 0.31
#_________________________________________________________marketData___Accuracy = 0.51, Precision = 0.26, Recall = 0.27
gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

# multi layer perceptron________________________________________set1___Accuracy = 0.47, Precision = 0.23, Recall = 0.25
#_______________________________________________________________set2___Accuracy = 0.51, Precision = 0.25, Recall = 0.27
#_______________________________________________________________set3___Accuracy = 0.53, Precision = 0.27, Recall = 0.28
#_______________________________________________________________set4___Accuracy = 0.47, Precision = 0.23, Recall = 0.24
#____________________________________________________fundamentalData___Accuracy = 0.36, Precision = 0.24, Recall = 0.24
#_________________________________________________________marketData___Accuracy = 0.51, Precision = 0.26, Recall = 0.27
mlp = MLPClassifier(hidden_layer_sizes=(5,7), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#### validate
predicted = cross_validation.cross_val_predict(sgdc, marketData, outputLabels, cv=4)
print("Accuracy: %0.2f" % (metrics.accuracy_score(outputLabels, predicted)))
print("Precision: %0.2f" % (metrics.precision_score(outputLabels, predicted, average='macro', sample_weight=None)))
print("Recall: %0.2f" % (metrics.recall_score(outputLabels, predicted, average='macro', sample_weight=None)))


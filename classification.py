import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect

from sklearn import preprocessing, linear_model, metrics, svm, neighbors, cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree


f = open('classes.txt', 'w').close()
f = open('classes.txt', 'a')

##### upload data
data = pd.read_csv("model_data.csv", sep=';', header = 0);

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

# make arrays
fundamentalData.values
marketData.values
allData.values
set1 = set1.values
set2 = set2.values
output = output.values.ravel()

# split data into bins
### with bisect
labels = ['strong see', 'sell', 'hold', 'buy', 'strong buy', 'very strong buy']
breakpoints = [1, 2, 3, 4, 5]
def label(total):
	return labels[bisect(breakpoints, total)]
outputLabels = map(label, output)
f.write ('LEGEND: \n 0-1:A (strong sell) \n 1-2:B (sell) \n 2-3:C (hold) \n 3-4:D (buy)\n 4-5:E (strong buy) \n 5:F (very strong buy) \n \n')

### with pandas cut -- splitting is relative to data values - does not seem suited
#outputLabels = pd.cut(output, 5, right=True, labels=['strong sell', 'sell', 'hold', 'buy', 'strong buy' ], retbins=True, precision=3, include_lowest=True)
#outputLabels = np.asarray(outputLabels)

# validate bins labels in 'classes.txt' file
for i in xrange(1,len(output)):
	f.write(str(output[i]) + '      ' + str(outputLabels[i]) + '\n')

#### classify

# linear SVC __________________________________________________________Accuracy = 0.45, Precision = 0.25, Recall = 0.24
lsvr = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

# Stochastic Gradient Descent with one-vs-all multiclassification _____Accuracy = 0.47, Precision = 0.26, Recall = 0.26
sgdc = linear_model.SGDClassifier(loss='perceptron', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, n_iter=None)

# Multi class decision tree  __________________________________________Accuracy = 0.47, Precision = 0.28, Recall = 0.29
dtc = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

#### validate
predicted = cross_validation.cross_val_predict(dtc, set2, outputLabels, cv=10)
print("Accuracy: %0.2f" % (metrics.accuracy_score(outputLabels, predicted)))
print("Precision: %0.2f" % (metrics.precision_score(outputLabels, predicted, average='macro', sample_weight=None)))
print("Recall: %0.2f" % (metrics.recall_score(outputLabels, predicted, average='macro', sample_weight=None)))


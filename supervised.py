import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, linear_model, metrics, svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

##### files for output
f = open('output.txt', 'w').close()
f = open('output.txt', 'a')

##### upload data
data = pd.read_csv("model_data.csv", sep=';', header = 0);

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

set1 = data [['operational cash flow', 'P/E', 'EPS', 'adjusted beta', 'return last 3 month']]
set1.loc[:, 'operational cash flow'] = set1.loc[:,'operational cash flow'] / fund_data.loc[:,'market cap']
set2 = data [['volatility 360 days', 'adjusted beta', 'returns last 6 months']]

header = set1.columns.values
print header.shape
npArray = np.array(set1)
npheader = np.array(header[1:-1])
print("Array shape X = %d, Y = %d " % (npArray.shape))
print (npArray)
npArray = npArray.astype(float)

output_en = np.asarray(output['analyst rating'].values, dtype="|S6")

##### create bins for output variable
#bins = [1, 2,]

##### visualize
## TODO gradient boosting / random forrest

# styling of graphs
colMap={0:"red", 1:"blue", 2:"yellow", 3:"green", 4:"blue", 5:"white", 6:"black", 7:"brown", 8:"pink"}
#plt.style.use('ggplot')

# individual histograms
#fund_data.hist()
#plt.show()
## DATA NOTES: data above is very skewed to the left
#mark_data.hist()
#plt.show()
## DATA NOTES: data fairly even distributed, exceptions skewed to the left: volatility 30 days, 90 days, 360 days, return 5 years, returns last year


#pp = sns.pairplot(all_int_data, x_vars = ['quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'P/E', 'EPS', 'market cap', 'total assets', 'number of employees', 'raw beta on SPX market', 'adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'returns last 5 years'], y_vars = ["analyst rating"])
#plt.show()

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
#pd.plotting.scatter_matrix(fund_data, alpha=0.7, diagonal = 'hist', figsize=[8,8], s=100)
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

# encode y for some classifiers
#en = preprocessing.LabelEncoder()
#en.fit(output.values)
#output_en = output['analyst rating'].values.ravel()


# ordinary least squares ___Accuracy -0.02 on fund_data, 0.02 on market data (unscaled)
reg = linear_model.LinearRegression()
# logistic regression ______Accuracy 0.07 on fund_data, 0.09 on market data (unscaled)
l_reg = linear_model.LogisticRegressionCV()
# lasso ____________________Accuracy 0.04 on fund_data
lss = linear_model.Lasso(alpha=0.1)
# linear kernel SVM ________Running does not finish
svm = svm.SVC(kernel='linear', C=1)

# cross validation
predicted = cross_val_score(reg, npArray, output_en, cv=5)

#print("Accuracy: %0.2f (+/- %0.2f)" % (predicted.mean(), predicted.std() * 2))
#print("Accuracy: " + metrics.accuracy_score(set1, predicted))
#print("Classification report \n")
#print metrics.classification_report(output, predicted)

#clf = MLPClassifier (solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(2,2), random_state=1)
#clf.fit(mark_data, y)

#out = clf.predict ([[1, 10, 63915, 55689, 24733, 20196, 31, 2],[0,	2, 59387, 50209, 5400, 8222, 21, 6]] )

#print(out)


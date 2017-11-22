import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

##### upload data

data = pd.read_csv("model_data.csv", sep=';', header = 0);

#data = data.applymap(str);
data = data.replace({'%':''}, regex=True);

##### split data into data frames

fund_data = data[['quick ratio', 'inventory turnover', 'sales revenue turnover', 'gross profit', 'net income', 'operational cash flow', 'P/E', 'EPS']]
mark_data = data[['raw beta on INDU market', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'returns last 5 years']]
output = data[['analyst rating']]

print('\n fundamental analysis data..... type: ' + str(type(fund_data)) + '\n')
#print(fund_data.head())

print('\n market data...type: ' + str(type(mark_data)) + '\n')
print(mark_data.head())

print('\n output data....type: ' + str(type(output)) + '\n')
print(output.head())

##### preprocessing

mark_scaled = preprocessing.scale(mark_data)
#print (mark_scaled);

##### visualize data

colMap={0:"red", 1:"blue", 2:"yellow", 3:"green", 4:"blue", 5:"white", 6:"black", 7:"brown", 8:"pink"}
plt.style.use('ggplot')
pd.plotting.scatter_matrix(mark_data, alpha=0.7, diagonal = 'hist', figsize=[8,8], s=100)
#plt.show()

##### classify

X = mark_data.values
y = output.values.ravel()

clf = MLPClassifier (solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(2,2), random_state=1)
clf.fit(mark_data, y)

out = clf.predict ([[1, 10, 63915, 55689, 24733, 20196, 31, 2],[0,	2, 59387, 50209, 5400, 8222, 21, 6]] )
print('\n prediction')
print(out)


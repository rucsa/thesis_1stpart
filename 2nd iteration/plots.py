from dataProcessing import nov_processed, y_class
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bisect import bisect
from itertools import combinations
import utils as util
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

y = y_class #encoded rating into label
y = pd.DataFrame(y_class)
X = nov_processed.drop(['analyst rating'], axis=1)
X = pd.concat([X, y], axis=1) #add the labels
'''
ss = X.loc[X['analyst rating'] == 'strong sell']
s = X.loc[X['analyst rating'] == 'sell']
h = X.loc[X['analyst rating'] == 'hold']
b = X.loc[X['analyst rating'] == 'buy']
sb = X.loc[X['analyst rating'] == 'strong buy']
'''

ss = X.loc[X['analyst rating'] == 1]
s = X.loc[X['analyst rating'] == 2]
h = X.loc[X['analyst rating'] == 3]
b = X.loc[X['analyst rating'] == 4]
sb = X.loc[X['analyst rating'] == 5]

features = list(X.columns.values)
features.remove('analyst rating')
'''
for pair in combinations(features, 2):
    plt.figure(figsize=(7,7))
    plt.scatter(sb.loc[:, pair[0]], sb.loc[:, pair[1]], color='red', marker='o', label='strong buy')
    plt.scatter(b.loc[:, pair[0]], b.loc[:, pair[1]], color='green', marker='^', label='buy')
    plt.scatter(h.loc[:, pair[0]], h.loc[:, pair[1]], color='blue', marker='x', label='hold')
    plt.scatter(s.loc[:, pair[0]], s.loc[:, pair[1]], color='cyan', marker='v', label='sell')
    plt.scatter(ss.loc[:, pair[0]], ss.loc[:, pair[1]], color='black', marker='s', label='strong sell')
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.legend(loc='upper center')
    #plt.show()
    print('ploted {} vs {}'.format(pair[0], pair[1]))
'''
X = X.dropna(axis=0, how='any')
y = X.loc[:, 'analyst rating']
X = X.drop(['analyst rating'], axis=1)
svm = SVC(kernel='rbf', random_state=1, gamma=0.01, C=10)
X = X.as_matrix()
y = y.as_matrix()
svm.fit(X, y)
plot_decision_regions(X, y, svm) #get function from book
plt.legend(loc='upper left')
plt.show()
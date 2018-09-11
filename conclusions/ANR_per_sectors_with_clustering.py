# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:16:07 2018

@author: Ruxandra
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_hdf("dataWithOUTliers.hdf5")

fin_sector = data[data.Sector == 'Financial'].drop('Sector', axis = 1)
features = fin_sector.columns.values.tolist()
#features.remove('Sector')
i=1
folder = 'plots without outliers grouped'

correlated_features = ['Adjusted_beta', 'Return_last_year', 'PE', 'Market_cap', 'ANR']
fin_sector = fin_sector[correlated_features].reset_index().drop('index', axis=1)

#tune number of clusters
n_clusters = np.arange(2, 21, 1)
silhouette_results  = []
calinski_results = []
for n in n_clusters:
    model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
               precompute_distances='auto', verbose=0, random_state=None, copy_x=True, 
               n_jobs=1, algorithm='auto').fit(fin_sector)

    labels = model.labels_
    silhouette_results.append(metrics.silhouette_score(fin_sector, labels, metric='euclidean'))
    calinski_results.append(metrics.calinski_harabaz_score(fin_sector, labels))
    
line1, = plt.plot(n_clusters, silhouette_results, 'b', label='silhouette score')
line2, = plt.plot(n_clusters, calinski_results, 'r', label='calinski harabaz score')

plt.legend()

plt.ylabel('F1 score')
plt.xlabel('n_clusters')
plt.show()

# ideal clusters
n_clusters = 5
model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
               precompute_distances='auto', verbose=0, random_state=None, copy_x=True, 
               n_jobs=1, algorithm='auto').fit(fin_sector)
labels = model.labels_ 

fin_sector['Cluster'] = labels

y = fin_sector['ANR']
X = fin_sector.drop('ANR', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dtc = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=20, 
                               max_features=4, max_leaf_nodes=None,
                               min_impurity_decrease=1e-07, min_samples_leaf=0.45,
                               min_samples_split=0.2, min_weight_fraction_leaf=0.0,
                               presort=False, random_state=None, splitter='best').fit(X_train, y_train)
prediction = model.predict(X_test)

print("Sklearn given F1_score: {}".format(f1_score(y_test, prediction, labels = [1,2,3,4,5], average = 'micro')))
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:07:54 2018

@author: RuxandraV
"""
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans

import pandas as pd

# todo ORGANIZE to get one consistent table of results


data = pd.read_hdf("dataWithOUTliers.hdf5")

fin_sector = data[data.Sector == 'Financial']
features = fin_sector.columns.values.tolist()
features.remove('Sector')
i=1
folder = 'plots without outliers grouped'

correlated_features = ['Adjusted_beta', 'Return_last_year', 'PE', 'Market_cap', 'ANR']
fin_sector = fin_sector[correlated_features].reset_index().drop('index', axis=1)


if True:                                # try with and without poly features
    y = fin_sector['ANR']
    fin_sector = fin_sector.drop('ANR', axis = 1)
    poly = PolynomialFeatures(degree=3)
    fin_sector = poly.fit_transform(fin_sector)
    fin_sector = pd.DataFrame(fin_sector)
    fin_sector['ANR'] = y
    
if True:                                # add clusters feature
    n_clusters = 5
    model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                   precompute_distances='auto', verbose=0, random_state=None, copy_x=True, 
                   n_jobs=1, algorithm='auto').fit(fin_sector)
    labels = model.labels_ 
    fin_sector['Cluster'] = labels

y = fin_sector['ANR']
X = fin_sector.drop('ANR', axis = 1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

# base models
dtc = ('Decision Tree Classifier' , DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=20, 
                               max_features=4, max_leaf_nodes=None,
                               min_impurity_decrease=1e-07, min_samples_leaf=0.15,
                               min_samples_split=0.3, min_weight_fraction_leaf=0.0,
                               presort=False, random_state=None, splitter='best'))

ada_dtc = ('AdaBoost with base Decision Tree', AdaBoostClassifier(base_estimator=DecisionTreeClassifier
                                                                  (class_weight=None, criterion='entropy', max_depth=20, 
                                                                   max_features=5, max_leaf_nodes=None,
                                                                   min_impurity_decrease=1e-07, min_samples_leaf=0.15,
                                                                   min_samples_split=0.3, min_weight_fraction_leaf=0.0,
                                                                   presort=False, random_state=None, splitter='best'), 
                                                n_estimators=50, learning_rate=0.1, 
                                                algorithm='SAMME', random_state=42))

bag = ('Bagging with base Decision Tree', BaggingClassifier(base_estimator=DecisionTreeClassifier
                                                              (class_weight=None, criterion='entropy', max_depth=20, 
                                                               max_features=5, max_leaf_nodes=None,
                                                               min_impurity_decrease=1e-07, min_samples_leaf=0.15,
                                                               min_samples_split=0.3, min_weight_fraction_leaf=0.0,
                                                               presort=False, random_state=None, splitter='best'), 
                                                n_estimators=50, max_samples=1.0, 
                                                max_features=1.0, bootstrap=True, bootstrap_features=True, 
                                                oob_score=False, warm_start=False, n_jobs=1, random_state=None, 
                                                verbose=0))

svc = ('SVM', SVC(C=1, kernel='linear', degree=2, gamma='auto', coef0=0.0, shrinking=True, 
                                    probability=False, tol=0.001, cache_size=200, class_weight='balanced', 
                                    verbose=False, max_iter=-1, decision_function_shape='ovr', 
                                    random_state=None))
  
ovr = ('OneVSRest MultiClassifier with base SVM', OneVsRestClassifier(svc[1]))

experiments = [dtc, ada_dtc, bag, svc, ovr] 
for model in experiments:
    model[1].fit(X_train, y_train)
    prediction = model[1].predict(X_test)
    print("Model {} F1_score: {}".format(model[0], f1_score(y_test, prediction, labels = [1,2,3,4,5], average = 'micro')))
#
#def svc_param_selection(X, y, nfolds):
#    Cs = [0.001, 0.01, 0.1, 1, 10]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#
#print(svc_param_selection(X, y, 4))

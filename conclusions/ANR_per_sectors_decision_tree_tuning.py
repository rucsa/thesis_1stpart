# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:09:36 2018

@author: RuxandraV
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier



from sklearn.metrics import f1_score

data = pd.read_hdf("dataWithOUTliers.hdf5")

fin_sector = data[data.Sector == 'Financial']
features = fin_sector.columns.values.tolist()
features.remove('Sector')
i=1
folder = 'plots without outliers grouped'

correlated_features = ['Adjusted_beta', 'Return_last_year', 'PE', 'Market_cap', 'ANR']
fin_sector = fin_sector[correlated_features].reset_index().drop('index', axis=1)

y = fin_sector['ANR']
X = fin_sector.drop('ANR', axis = 1)

poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# tune max_depth
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    dt = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',max_depth=max_depth)
    dt.fit(X_train, y_train)
    
    train_pred = dt.predict(X_train)
    train_results.append(f1_score(y_train, train_pred, labels = [1,2,3,4,5], average = 'micro'))
       
    y_pred = dt.predict(X_test)
    test_results.append(f1_score(y_test, y_pred, labels = [1,2,3,4,5], average = 'micro'))
    

line1 = plt.plot(max_depths, train_results, 'b', label='Train F1')
line2 = plt.plot(max_depths, test_results, 'r', label='Test F1')

plt.legend()

plt.ylabel('F1 score')
plt.xlabel('Tree depth')
plt.show()
   
# tune min_sample_split
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', min_samples_split=min_samples_split)
   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)
   train_results.append(f1_score(y_train, train_pred, labels = [1,2,3,4,5], average = 'micro'))
       
   y_pred = dt.predict(X_test)
   test_results.append(f1_score(y_test, y_pred, labels = [1,2,3,4,5], average = 'micro'))

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train F1')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test F1')

plt.legend()

plt.ylabel('F1 score')
plt.xlabel('min samples split')
plt.show()
   
# tune min_sample_leaf
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', min_samples_leaf=min_samples_leaf)
   dt.fit(X_train, y_train)
   
   train_pred = dt.predict(X_train)
   train_results.append(f1_score(y_train, train_pred, labels = [1,2,3,4,5], average = 'micro'))
       
   y_pred = dt.predict(X_test)
   test_results.append(f1_score(y_test, y_pred, labels = [1,2,3,4,5], average = 'micro'))

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train F1')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test F1')

plt.legend()

plt.ylabel('F1 score')
plt.xlabel('min samples leaf')
plt.show()

# tune max_feature

max_features = list(range(1, X_train.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
   dt = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_features=max_feature)
   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)
   train_results.append(f1_score(y_train, train_pred, labels = [1,2,3,4,5], average = 'micro'))
       
   y_pred = dt.predict(X_test)
   test_results.append(f1_score(y_test, y_pred, labels = [1,2,3,4,5], average = 'micro'))

line1, = plt.plot(max_features, train_results, 'b', label='Train F1')
line2, = plt.plot(max_features, test_results, 'r', label='Test F1')

plt.legend()

plt.ylabel('F1 score')
plt.xlabel('max features')
plt.show()

# make prediction

dtc = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=20, 
                               max_features=4, max_leaf_nodes=None,
                               min_impurity_decrease=1e-07, min_samples_leaf=0.15,
                               min_samples_split=0.3, min_weight_fraction_leaf=0.0,
                               presort=False, random_state=None, splitter='best')#.fit(X_train, y_train)

model = AdaBoostClassifier(base_estimator=dtc, n_estimators=50, learning_rate=0.1, 
                           algorithm='SAMME', random_state=42).fit(X_train, y_train)
  
model = BaggingClassifier(base_estimator=dtc, n_estimators=50, max_samples=1.0, 
                          max_features=1.0, bootstrap=True, bootstrap_features=True, 
                          oob_score=False, warm_start=False, n_jobs=1, random_state=None, 
                          verbose=0).fit(X_train, y_train)

predictions = model.predict(X_test)

bench1 = []
for i in range(0, len(predictions)):
    bench1.append(1)
    
bench2 = []
for i in range(0, len(predictions)):
    bench2.append(2)
    
bench3 = []
for i in range(0, len(predictions)):
    bench3.append(3)
    
bench4 = []
for i in range(0, len(predictions)):
    bench4.append(4)
    
bench5 = []
for i in range(0, len(predictions)):
    bench5.append(5)
    
average_b = (f1_score(y_test, bench1, labels = [1,2,3,4,5], average = 'micro') +
            f1_score(y_test, bench2, labels = [1,2,3,4,5], average = 'micro') + 
            f1_score(y_test, bench3, labels = [1,2,3,4,5], average = 'micro') + 
            f1_score(y_test, bench4, labels = [1,2,3,4,5], average = 'micro') + 
            f1_score(y_test, bench5, labels = [1,2,3,4,5], average = 'micro'))/5


print("Sklearn given F1_score: {}".format(f1_score(y_test, predictions, labels = [1,2,3,4,5], average = 'micro')))

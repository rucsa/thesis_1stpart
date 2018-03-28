import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import prediction as predict
import utils as util
import dataProcessing as dp

from scipy import stats
import collections


from itertools import combinations

X = pd.read_hdf('data.hdf5', 'Datataset1/X')
y = X.loc[:,['analyst rating']]
X = X.drop(['analyst rating'], axis=1)

features = list(X.columns.values)


writer = pd.ExcelWriter('../results/results2.xlsx', engine='xlsxwriter')
lreg = LinearRegression()
results = {}
count = 0
for i in range (1, 2):
    print ('computing {} features'.format(i))
    for feature_list in combinations(features, 10):
        feature_list = list(feature_list)
        feature_list.append('analyst rating')
        #X = X.loc[:, feature_list].dropna()
        #y = X.loc[:, 'analyst rating']
        #X = X.drop(['analyst rating'], axis=1)
        print ('computing {}'.format(feature_list))
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        lreg.fit(X_train, y_train)
        predicted = lreg.predict(X_test)

        residuals = y_test - predicted
        mean_obs = np.mean(y_test)
        ss_res = np.sum((y_test - predicted)**2)
        ss_tot = np.sum((y_test - mean_obs)**2)
        mse = np.mean((predicted - y_test)**2)
        r = lreg.score(X_test, y_test)
        r2 = 1-(ss_res/ss_tot)
        #print ('Mean squared error = {}'.format(mse))
        #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))
        feature_list.remove('analyst rating')
        key = [feature_list]
        results[float(r2)] = key
        count = count + 1
    print ('finished computing {} features'.format(i))

sorted_dict = collections.OrderedDict(sorted(results.items(), reverse=True))
df = pd.DataFrame.from_dict(sorted_dict, orient='index')
df.to_excel(writer, sheet_name='Sheet3')
#for k, v in sorted_dict.items():
#    print('{} : {}'.format(k, v))
#    output.to_excel(writer, sheet_name='Sheet3')

util.count_duplicates(count, results)

writer.save()
writer.close()
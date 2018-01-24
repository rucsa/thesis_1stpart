import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

import prediction as predict
import utils as util
import dataProcessing as dp

from scipy import stats
import collections
import math

'''' upload data '''
data_nov = pd.read_excel('../../Data/BLB_data_only_values_1511.xlsx')
data_dec = pd.read_excel('../../Data/BLB_data_only_values_1512.xlsx')
data_jan = pd.read_excel('../../Data/BLB_data_only_values_1501.xlsx')

''' extract features that express trend movement (substracting one month from another doesn't bring the whole column to zero)'''
numeric_nov = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]
numeric_dec = data_dec[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]
numeric_jan = data_jan[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap']]

numeric_nov_all = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'total assets', 'analyst rating']]
numeric_nov_all_no_anr = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'total assets']]
list = ['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover', 'sale ravenue turnover', 'gross profit', 'net income', 'operational cash flow', 'total assets']
results = []

''' plot linear regression trendline '''
''' individual feature correlation '''
for item in list:
    p = numeric_nov_all_no_anr[[item]].dropna().values
    #min = np.percentile(p, 25)
    max = np.percentile(p, 75)
    #print '%s values should be between %s and %s' % (item, 0, max)
    pe = data_nov[[item, 'analyst rating']]
    pe = pe[np.isfinite(pe[item])]
    pe = pe.loc[pe[item] <= max]
    anr = pe[['analyst rating']].values
    pe = pe[[item]].values
    #fig = plt.figure()
    #plt.ylabel('Analyst rating')
    #plt.xlabel(item)
    # linregress needs one dimentional array - obtained through slicing
    #plt.scatter(pe, anr)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(pe[:,0], anr[:,0])
    #plt.plot(pe, intercept + slope*pe, 'r', label='fitted line')
    #plt.show()
    nl = '\n'
    results.append("{} has correlation of {} and R-squared: {}\n".format(item, rvalue, rvalue**2))

for result in results:
    print (result)
print (numeric_nov_all.corr()['analyst rating'])
print ("\n")

results = {}
''' 2 features correlations '''
lreg = LinearRegression()
count2 = 0
for i in range(0, len(list)-1):
    for j in range(i+1, len(list)):
        X = numeric_nov_all.loc[:, [list[i], list[j], 'analyst rating']].dropna()
        y = X.loc[:, 'analyst rating']
        X = X.drop(['analyst rating'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        #print('Using features: {} and {} '.format(list[i], list[j]))

        lreg.fit(X_train, y_train)
        predicted = lreg.predict(X_test)

        residuals = y_test - predicted
        mean_obs = np.mean(y_test)
        ss_res = np.sum((y_test - predicted)**2)
        ss_tot = np.sum((y_test - mean_obs)**2)
        mse = np.mean((predicted - y_test)**2)
        r = lreg.score(X_test, y_test)
        #print ('Mean squared error = {}'.format(mse))
        #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))

        key = [list[i], list[j]]
        results[r] = key
        count2 = count2 + 1

''' 3 features correlation '''
count3 = 0
for i in range(0, len(list)-2):
    for j in range(i+1, len(list)-1):
        for k in range (j+1, len(list)):
            X = numeric_nov_all.loc[:, [list[i], list[j], list[k], 'analyst rating']].dropna()
            y = X.loc[:, 'analyst rating']
            X = X.drop(['analyst rating'], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            #print('Using features: {}, {} and {} '.format(list[i], list[j], list[k]))

            lreg.fit(X_train, y_train)
            predicted = lreg.predict(X_test)

            residuals = y_test - predicted
            mean_obs = np.mean(y_test)
            ss_res = np.sum((y_test - predicted)**2)
            ss_tot = np.sum((y_test - mean_obs)**2)
            mse = np.mean((predicted - y_test)**2)
            r = lreg.score(X_test, y_test)

            #print ('Mean squared error = {}'.format(mse))
            #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))

            key = [list[i], list[j], list[k]]
            results[r] = key
            count3 = count3 + 1

''' 4 features correlation '''
count4 = 0
for i in range(0, len(list)-3):
    for j in range(i+1, len(list)-2):
        for k in range (j+1, len(list)-1):
            for l in range (k+1, len(list)):
                X = numeric_nov_all.loc[:, [list[i], list[j], list[k], list[l], 'analyst rating']].dropna()
                y = X.loc[:, 'analyst rating']
                X = X.drop(['analyst rating'], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y)
                #print('Using features: {}, {} and {} '.format(list[i], list[j], list[k]))

                lreg.fit(X_train, y_train)
                predicted = lreg.predict(X_test)

                residuals = y_test - predicted
                mean_obs = np.mean(y_test)
                ss_res = np.sum((y_test - predicted)**2)
                ss_tot = np.sum((y_test - mean_obs)**2)
                mse = np.mean((predicted - y_test)**2)
                r = lreg.score(X_test, y_test)

                #print ('Mean squared error = {}'.format(mse))
                #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))

                key = [list[i], list[j], list[k], list[l]]
                results[r] = key
                count4 = count4 + 1

''' 5 features correlation '''
count5 = 0
for i in range(0, len(list)-4):
    for j in range(i+1, len(list)-3):
        for k in range (j+1, len(list)-2):
            for l in range (k+1, len(list)-1):
                for m in range (l+1, len(list)):
                    X = numeric_nov_all.loc[:, [list[i], list[j], list[k], list[l], list[m], 'analyst rating']].dropna()
                    y = X.loc[:, 'analyst rating']
                    X = X.drop(['analyst rating'], axis=1)
                    X_train, X_test, y_train, y_test = train_test_split(X, y)
                    #print('Using features: {}, {} and {} '.format(list[i], list[j], list[k]))

                    lreg.fit(X_train, y_train)
                    predicted = lreg.predict(X_test)

                    residuals = y_test - predicted
                    mean_obs = np.mean(y_test)
                    ss_res = np.sum((y_test - predicted)**2)
                    ss_tot = np.sum((y_test - mean_obs)**2)
                    mse = np.mean((predicted - y_test)**2)
                    r = lreg.score(X_test, y_test)

                    #print ('Mean squared error = {}'.format(mse))
                    #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))

                    key = [list[i], list[j], list[k], list[l], list[m]]
                    results[r] = key
                    count5 = count5 + 1

''' 6 features correlation '''
count6 = 0
for i in range(0, len(list)-5):
    for j in range(i+1, len(list)-4):
        for k in range (j+1, len(list)-3):
            for l in range (k+1, len(list)-2):
                for m in range (l+1, len(list)-1):
                    for n in range (m+1, len(list)):
                        X = numeric_nov_all.loc[:, [list[i], list[j], list[k], list[l], list[m], list[n], 'analyst rating']].dropna()
                        y = X.loc[:, 'analyst rating']
                        X = X.drop(['analyst rating'], axis=1)
                        X_train, X_test, y_train, y_test = train_test_split(X, y)
                        #print('Using features: {}, {} and {} '.format(list[i], list[j], list[k]))

                        lreg.fit(X_train, y_train)
                        predicted = lreg.predict(X_test)

                        residuals = y_test - predicted
                        mean_obs = np.mean(y_test)
                        ss_res = np.sum((y_test - predicted)**2)
                        ss_tot = np.sum((y_test - mean_obs)**2)
                        mse = np.mean((predicted - y_test)**2)
                        r = lreg.score(X_test, y_test)

                        #print ('Mean squared error = {}'.format(mse))
                        #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))

                        key = [list[i], list[j], list[k], list[l], list[m], list[n]]
                        results[r] = key
                        count6 = count6 + 1
                        
''' 7 features correlation '''
count7 = 0
for i in range(0, len(list)-6):
    for j in range(i+1, len(list)-5):
        for k in range (j+1, len(list)-4):
            for l in range (k+1, len(list)-3):
                for m in range (l+1, len(list)-2):
                    for n in range (m+1, len(list)-1):
                        for o in range (n+1, len(list)):
                            X = numeric_nov_all.loc[:, [list[i], list[j], list[k], list[l], list[m], list[n], list[o], 'analyst rating']].dropna()
                            y = X.loc[:, 'analyst rating']
                            X = X.drop(['analyst rating'], axis=1)
                            X_train, X_test, y_train, y_test = train_test_split(X, y)
                            #print('Using features: {}, {} and {} '.format(list[i], list[j], list[k]))
    
                            lreg.fit(X_train, y_train)
                            predicted = lreg.predict(X_test)
    
                            residuals = y_test - predicted
                            mean_obs = np.mean(y_test)
                            ss_res = np.sum((y_test - predicted)**2)
                            ss_tot = np.sum((y_test - mean_obs)**2)
                            mse = np.mean((predicted - y_test)**2)
                            r = lreg.score(X_test, y_test)
    
                            #print ('Mean squared error = {}'.format(mse))
                            #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))
    
                            key = [list[i], list[j], list[k], list[l], list[m], list[n], list[o]]
                            results[r] = key
                            count7 = count7 + 1
                            
''' 8 features correlation '''
count8 = 0
for i in range(0, len(list)-7):
    for j in range(i+1, len(list)-6):
        for k in range (j+1, len(list)-5):
            for l in range (k+1, len(list)-4):
                for m in range (l+1, len(list)-3):
                    for n in range (m+1, len(list)-2):
                        for o in range (n+1, len(list)-1):
                            for p in range (o+1, len(list)):
                                X = numeric_nov_all.loc[:, [list[i], list[j], list[k], list[l], list[m], list[n], list[o], list[p], 'analyst rating']].dropna()
                                y = X.loc[:, 'analyst rating']
                                X = X.drop(['analyst rating'], axis=1)
                                X_train, X_test, y_train, y_test = train_test_split(X, y)
                                #print('Using features: {}, {} and {} '.format(list[i], list[j], list[k]))
        
                                lreg.fit(X_train, y_train)
                                predicted = lreg.predict(X_test)
        
                                residuals = y_test - predicted
                                mean_obs = np.mean(y_test)
                                ss_res = np.sum((y_test - predicted)**2)
                                ss_tot = np.sum((y_test - mean_obs)**2)
                                mse = np.mean((predicted - y_test)**2)
                                r = lreg.score(X_test, y_test)
        
                                #print ('Mean squared error = {}'.format(mse))
                                #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))
        
                                key = [list[i], list[j], list[k], list[l], list[m], list[n], list[o], list[p]]
                                results[r] = key
                                count8 = count8 + 1

sorted = collections.OrderedDict(sorted(results.items()))
for k, v in sorted.items():
    print('{} : {}'.format(k, v))

util.count_duplicates(count2 + count3 + count4 + count5 + count6 + count7 + count8, results)

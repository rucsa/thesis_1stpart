import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm


X = pd.read_hdf('data.hdf5', 'Datataset1/X')
anr = X['analyst rating']
#data = X[['return last year', 'P/E', 'volatility 360 days', 'analyst rating']]
#data = X[['return last year', 'volatility 360 days', 'analyst rating']]
data = X

#for feature in data:
#    plt.scatter(X[feature], anr, s=7)
#    plt.ylabel('ANR')
#    plt.xlabel(feature)
#    plt.show()
#    
data = data.dropna(axis=0, how='any')

''' remove outliers'''


remove_outliers = True


if remove_outliers:
    data = data.loc[data['adjusted beta'] >= -4]
    data = data.loc[data['adjusted beta'] <= 4]
    
    data = data.loc[data['volatility 30 days'] >= -1.8]
    data = data.loc[data['volatility 30 days'] <= 6]
    
    data = data.loc[data['volatility 90 days'] >= -1.9]
    data = data.loc[data['volatility 90 days'] <= 4.5]
    
    data = data.loc[data['volatility 360 days'] >= -2]
    data = data.loc[data['volatility 360 days'] <= 3.5]
    
    data = data.loc[data['return last 3 month'] >= -2.5]
    data = data.loc[data['return last 3 month'] <= 4.3]
    
    data = data.loc[data['returns last 6 months'] >= -2.5]
    data = data.loc[data['returns last 6 months'] <= 4]
    
    data = data.loc[data['return last year'] >= -2.5]
    data = data.loc[data['return last year'] <= 5.5]
    
    data = data.loc[data['returns last 5 years'] >= -3]
    data = data.loc[data['returns last 5 years'] <= 10]
    
    data = data.loc[data['P/E'] >= -0.5]
    data = data.loc[data['P/E'] <= 15]
    
    data = data.loc[data['EPS'] >= -4.5]
    data = data.loc[data['EPS'] <= 10]

    data = data.loc[data['market cap'] <= 10.5]
    
    data = data.loc[data['quick ratio'] <= 7.8]
    
    data = data.loc[data['inventory turnover'] >= -0.5]
    data = data.loc[data['inventory turnover'] <= 4.8]
    
    data = data.loc[data['revenu'] >= -1]
    data = data.loc[data['revenu'] <= 10]
    
    data = data.loc[data['gross profit'] >= -1]
    data = data.loc[data['gross profit'] <= 8]
    
    data = data.loc[data['net income'] >= -5]
    data = data.loc[data['net income'] <= 5]
    
    data = data.loc[data['operational cash flow'] >= -2.5]
    data = data.loc[data['operational cash flow'] <= 5]
    
    data = data.loc[data['total assets'] >= -0.5]
    data = data.loc[data['total assets'] <= 4]
    
    data = data.loc[data['PSR'] >= -0.5]
    data = data.loc[data['PSR'] <= 6]
    
    data.to_hdf('dataWithOUTliers.hdf5', 'Datataset1/X')


#data = data.loc[data['volatility 360 days'] <= 35]
#data = data.loc[data['volatility 360 days'] >= 14]

#print(data.describe())
#print(data.corr())
#
#text_file = open("Output {}.txt".format('4'), "w")
#
#features = data.columns.values.tolist()
#feat = []
##for item in features:
##    if (item == 'analyst rating'):
##        break
##    feat.append(item)
#exog = sm.add_constant(data[['return last year', 'adjusted beta', 'market cap']], prepend=False)
## Fit and summarize OLS model
#mod = sm.OLS(data['analyst rating'], exog)
#res = mod.fit()
#print(res.summary())
#text_file.write(res.summary().as_text())
#
#text_file.close()

anr = data['analyst rating']
data = data.drop(['analyst rating'], axis=1)
features = data.columns.values.tolist()
i=1

if remove_outliers:
    folder = 'plots without outliers'
else:
    folder = 'plots with outliers'
    
for item in features:
    p = data[[item]].dropna().values
    #min = np.percentile(p, 25)
    #max = np.percentile(p, 75)
    #print '%s values should be between %s and %s' % (item, 0, max)
    pe = pd.concat([data[[item]], pd.DataFrame(anr, columns=['analyst rating'])], axis=1)
    pe = pe[np.isfinite(pe[item])]
    #pe = pe.loc[pe[item] <= max]
    #pe = pe.loc[pe[item] >= min]
    anr_slice = pe[['analyst rating']].values
    pe = pe[[item]].values
    fig = plt.figure(figsize=(7,7))
    plt.ylabel('Analyst rating')
    plt.xlabel(item)
    # linregress needs one dimentional array - obtained through slicing
    plt.scatter(pe, anr_slice, s=7)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(pe[:,0], anr_slice[:,0])
    plt.plot(pe, intercept + slope*pe, 'r', label='fitted line')
    file_name = folder + '/anr_feature' + str(i)  + '.png'
    i=i+1
    plt.savefig(file_name, format='png')

plt.close()
    

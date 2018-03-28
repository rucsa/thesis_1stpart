import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataProcessing import y
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


data = data.loc[data['return last year'] <= 30]
data = data.loc[data['return last year'] >= -20]
data = data.loc[data['adjusted beta'] <= 2]
data = data.loc[data['adjusted beta'] >= -2]
data = data.loc[data['market cap'] <= 14000]
data = data.loc[data['market cap'] >= 1000]

##data = data.loc[data['P/E'] <= 40]
##data = data.loc[data['P/E'] >= 5]
#data = data.loc[data['volatility 360 days'] <= 35]
#data = data.loc[data['volatility 360 days'] >= 14]

print(data.describe())
print(data.corr())

text_file = open("Output {}.txt".format('4'), "w")

features = data.columns.values.tolist()
feat = []
#for item in features:
#    if (item == 'analyst rating'):
#        break
#    feat.append(item)
exog = sm.add_constant(data[['return last year', 'adjusted beta', 'market cap']], prepend=False)
# Fit and summarize OLS model
mod = sm.OLS(data['analyst rating'], exog)
res = mod.fit()
print(res.summary())
text_file.write(res.summary().as_text())

text_file.close()

#anr = data['analyst rating']
#data = data.drop(['analyst rating'], axis=1)
#features = data.columns.values.tolist()
#i=1
#for item in features:
#    p = data[[item]].dropna().values
#    min = np.percentile(p, 25)
#    max = np.percentile(p, 75)
#    #print '%s values should be between %s and %s' % (item, 0, max)
#    pe = pd.concat([data[[item]], pd.DataFrame(anr, columns=['analyst rating'])], axis=1)
#    pe = pe[np.isfinite(pe[item])]
#    #pe = pe.loc[pe[item] <= max]
#    #pe = pe.loc[pe[item] >= min]
#    anr_slice = pe[['analyst rating']].values
#    pe = pe[[item]].values
#    fig = plt.figure()
#    plt.ylabel('Analyst rating')
#    plt.xlabel(item + " " + str(anr_slice.size) + " elements")
#    # linregress needs one dimentional array - obtained through slicing
#    plt.scatter(pe, anr_slice, s=7)
#    slope, intercept, rvalue, pvalue, stderr = stats.linregress(pe[:,0], anr_slice[:,0])
#    plt.plot(pe, intercept + slope*pe, 'r', label='fitted line')
#    file_name = 'plots/anr_feature' + str(i)  + '.png'
#    i=i+1
#    plt.savefig(file_name, format='png')
#
#plt.close()
    

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm


X = pd.read_hdf('data.hdf5', 'Datataset1/X', format='table')
anr = X['ANR']
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
    data = data.loc[data['Adjusted_beta'] >= -4]
    data = data.loc[data['Adjusted_beta'] <= 4]
    
    data = data.loc[data['Volatility_30'] >= -1.8]
    data = data.loc[data['Volatility_30'] <= 6]
    
    data = data.loc[data['Volatility_90'] >= -1.9]
    data = data.loc[data['Volatility_90'] <= 4.5]
    
    data = data.loc[data['Volatility_360'] >= -2]
    data = data.loc[data['Volatility_360'] <= 3.5]
    
    data = data.loc[data['Returns_3_months'] >= -2.5]
    data = data.loc[data['Returns_3_months'] <= 4.3]
    
    data = data.loc[data['Returns_6_months'] >= -2.5]
    data = data.loc[data['Returns_6_months'] <= 4]
    
    data = data.loc[data['Return_last_year'] >= -2.5]
    data = data.loc[data['Return_last_year'] <= 5.5]
    
    data = data.loc[data['Returns_5_years'] >= -3]
    data = data.loc[data['Returns_5_years'] <= 10]
    
    data = data.loc[data['PE'] >= -0.5]
    data = data.loc[data['PE'] <= 15]
    
    data = data.loc[data['EPS'] >= -4.5]
    data = data.loc[data['EPS'] <= 10]

    data = data.loc[data['Market_cap'] <= 10.5]
    
    data = data.loc[data['Quick_ratio'] <= 7.8]
    
    data = data.loc[data['Inventory_turnover'] >= -0.5]
    data = data.loc[data['Inventory_turnover'] <= 4.8]
    
    data = data.loc[data['Revenue'] >= -1]
    data = data.loc[data['Revenue'] <= 10]
    
    data = data.loc[data['Gross_profit'] >= -1]
    data = data.loc[data['Gross_profit'] <= 8]
    
    data = data.loc[data['Net_income'] >= -5]
    data = data.loc[data['Net_income'] <= 5]
    
    data = data.loc[data['Operational_cash_flow'] >= -2.5]
    data = data.loc[data['Operational_cash_flow'] <= 5]
    
    data = data.loc[data['Assets'] >= -0.5]
    data = data.loc[data['Assets'] <= 4]
    
    data = data.loc[data['PSR'] >= -0.5]
    data = data.loc[data['PSR'] <= 6]
    
    data.to_hdf('dataWithOUTliers.hdf5', 'Datataset1/X', format = 'table')


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

anr = data['ANR']
data = data.drop(['ANR'], axis=1)
features = data.columns.values.tolist()
i=1

if remove_outliers:
    folder = 'plots without outliers'
else:
    folder = 'plots with outliers'
    
for item in features:
    if item != 'Sector':
        p = data[[item]].dropna().values
        #min = np.percentile(p, 25)
        #max = np.percentile(p, 75)
        #print '%s values should be between %s and %s' % (item, 0, max)
        pe = pd.concat([data[[item]], pd.DataFrame(anr, columns=['ANR'])], axis=1)
        pe = pe[np.isfinite(pe[item])]
        #pe = pe.loc[pe[item] <= max]
        #pe = pe.loc[pe[item] >= min]
        anr_slice = pe[['ANR']].values
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
    

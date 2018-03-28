import scipy as sy
import pandas as pd
import numpy as np
import utils as ut

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

def interpolate(dataFrame, method='values'):
    return dataFrame.interpolate(method=method)

def fill(dataFrame, method='', limit=1):
    if method=='':
        return dataFrame.fillna(0)
    else: return dataFrame.fillna(method=method, limit=limit)

def mask(dataFrame):
    return dataFrame.isnull()

def count_duplicates(counted, dictionary):
    if (counted == len(dictionary)):
        print ('no duplicate keys in dict')
    else:
        print ('{} elements have the same key'.format(len(dictionary) - counted))
    
def replace_nan(df):
    for i in range (0, df.size):
        if df[i].isnull:
            df[i] = 'N'
    return df


'''' upload data '''
data = pd.read_excel('../../Data/BLB_values_SP1500_22feb.xlsx')


numeric_all = data[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                            'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                            'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover',
                            'revenu', 'gross profit', 'net income', 'operational cash flow',
                            'total assets', 'analyst rating']]
numeric_all_no_anr = data[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                                   'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                                   'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover',
                                   'revenu', 'gross profit', 'net income', 'operational cash flow',
                                   'total assets']]


features = ['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month',
            'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years',
            'quick ratio', 'inventory turnover', 'revenu', 'gross profit', 'net income',
            'operational cash flow', 'total assets']

''' count nans per column '''
nans_n = numeric_all.isnull().sum()


''' scale out to market cap '''
nov_processed = numeric_all.copy()

feature_set = ['inventory turnover', 'revenu', 'gross profit', 
               'net income', 'operational cash flow',
                            'total assets']
nov_processed = ut.scaleOutMarketCap(nov_processed, feature_set)


''' create feature 'size' '''
nov_processed['size'] = ut.encodeMarketCap(nov_processed)

''' encode categorical variables '''
#string_nov = data[['Sector', 'region', 'ethics', 'bribery']]
#eth = data['ethics']
#eth = eth.apply(replace_nan)
#string_nov = ut.categoricalToNumeric(string_nov, LabelEncoder())
#nov_processed = pd.concat([nov_processed, string_nov], axis=1)

''' create feature PSR '''
nov_processed['PSR'] = nov_processed.loc[:,'market cap'] / nov_processed.loc[:,'revenu']
features = list(nov_processed.columns.values)


''' create y for regression'''
y = nov_processed.loc[:, 'analyst rating']
#nov_processed = nov_processed.drop(['analyst rating'], axis=1)

''' create y for classification'''
#y_class = pd.cut(y, bins=[0, 1.5, 2.5, 3.5, 4.5, 5], include_lowest=True, labels=['strong sell', 'sell', 'hold', 'buy', 'strong buy'])
# some classifiers don't take sttrings for classes
y_class = pd.cut(y, bins=[0, 1.5, 2.5, 3.5, 4.5, 5], include_lowest=True, labels=[1, 2, 3, 4, 5])
y_classs = pd.DataFrame(y_class)

''' drop ANR from X '''
nov_processed = nov_processed.fillna(0)
nov_processed = nov_processed.loc[nov_processed['analyst rating'] > 0]
y = nov_processed.loc[:, 'analyst rating']
nov_processed = nov_processed.drop(['analyst rating'], axis=1)

''' deal with missing values '''
#y_class = y_class.fillna(1)
#y = y.fillna(1)
#nov_processed = interpolate(nov_processed)
#imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp.fit_transform(nov_processed)

''' scale '''
if True:
    nov_processed.loc[:,'market cap'] = nov_processed.loc[:,'market cap']/1000
    nov_processed.loc[:,'revenu'] = nov_processed.loc[:,'revenu']/100
    nov_processed.loc[:,'gross profit'] = nov_processed.loc[:,'gross profit']/100
    nov_processed.loc[:,'net income'] = nov_processed.loc[:,'net income']/10
    nov_processed.loc[:,'operational cash flow'] = nov_processed.loc[:,'operational cash flow']/100
    nov_processed.loc[:,'total assets'] = nov_processed.loc[:,'total assets']/100
    
scaler = preprocessing.StandardScaler() # standard scaling
#scaler = preprocessing.MaxAbsScaler() # scale to range [0, 1]
#scaler = preprocessing.MaxAbsScaler() # scale to range [-1,1]
#scaler = preprocessing.RobustScaler() #scaling with outliers
#scaler = preprocessing.QuantileTransformer() #non-linear transformation
#scaler = preprocessing. Normalizer()
nov_processed = pd.DataFrame(scaler.fit_transform(nov_processed), columns=nov_processed.columns, index=nov_processed.index) 
nov_processed = pd.concat([nov_processed, y], axis='columns')
nov_processed.to_hdf('data.hdf5', 'Datataset1/X')


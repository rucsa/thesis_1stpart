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


'''' upload data '''
data_nov = pd.read_excel('../../Data/BLB_data_only_values_1511.xlsx')
data_dec = pd.read_excel('../../Data/BLB_data_only_values_1512.xlsx')
data_jan = pd.read_excel('../../Data/BLB_data_only_values_1501.xlsx')

numeric_nov = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                        'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                        'market cap']]
numeric_dec = data_dec[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                        'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                        'market cap']]
numeric_jan = data_jan[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                        'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                        'market cap']]

numeric_nov_all = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                            'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                            'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover',
                            'revenue', 'gross profit', 'net income', 'operational cash flow',
                            'total assets', 'analyst rating']]
numeric_nov_all_no_anr = data_nov[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                                   'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                                   'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover',
                                   'revenue', 'gross profit', 'net income', 'operational cash flow',
                                   'total assets']]

numeric_dec_all = data_dec[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                            'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                            'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover',
                            'revenue', 'gross profit', 'net income', 'operational cash flow',
                            'total assets', 'analyst rating']]
numeric_jan_all = data_jan[['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days',
                            'return last 3 month', 'returns last 6 months', 'return last year', 'P/E', 'EPS',
                            'market cap', 'returns last 5 years', 'quick ratio', 'inventory turnover',
                            'revenue', 'gross profit', 'net income', 'operational cash flow',
                            'total assets', 'analyst rating']]

features = ['adjusted beta', 'volatility 30 days', 'volatility 90 days', 'volatility 360 days', 'return last 3 month',
            'returns last 6 months', 'return last year', 'P/E', 'EPS', 'market cap', 'returns last 5 years',
            'quick ratio', 'inventory turnover', 'revenue', 'gross profit', 'net income',
            'operational cash flow', 'total assets']

''' count nans per column '''
nans_n = numeric_nov_all.isnull().sum()
nans_d = numeric_dec_all.isnull().sum()
nans_j = numeric_jan_all.isnull().sum()

''' scale out to market cap '''
nov_processed = numeric_nov_all.copy()
dec_processed = numeric_nov_all.copy()
jan_processed = numeric_nov_all.copy()
feature_set = ['inventory turnover', 'revenue', 'gross profit', 
               'net income', 'operational cash flow',
                            'total assets']
nov_processed = ut.scaleOutMarketCap(nov_processed, feature_set)
dec_processed = ut.scaleOutMarketCap(dec_processed, feature_set)
jan_processed = ut.scaleOutMarketCap(jan_processed, feature_set)

''' create feature 'size' '''
nov_processed['size'] = ut.encodeMarketCap(nov_processed)

''' encode categorical variables '''
string_nov = data_nov[['Sector', 'region', 'ethics', 'bribery']]
string_nov = ut.categoricalToNumeric(string_nov, LabelEncoder())
nov_processed = pd.concat([nov_processed, string_nov], axis=1)

''' create feature PSR '''
nov_processed['PSR'] = nov_processed.loc[:,'market cap'] / nov_processed.loc[:,'revenue']
features = list(nov_processed.columns.values)

''' create y for regression'''
y = nov_processed.loc[:, 'analyst rating']
nov_processed = nov_processed.drop(['analyst rating'], axis=1)

''' create y for classification'''
#y_class = pd.cut(y, bins=[0, 1.5, 2.5, 3.5, 4.5, 5], include_lowest=True, labels=['strong sell', 'sell', 'hold', 'buy', 'strong buy'])
# some classifiers don't take sttrings for classes
y_class = pd.cut(y, bins=[0, 1.5, 2.5, 3.5, 4.5, 5], include_lowest=True, labels=[1, 2, 3, 4, 5])

# deal with missing values
nov_processed = interpolate(nov_processed)
#imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp.fit_transform(nov_processed)

scaler = preprocessing.StandardScaler() # standard scaling
#scaler = preprocessing.MaxAbsScaler() # scale to range [0, 1]
#scaler = preprocessing.MaxAbsScaler() # scale to range [-1,1]
#scaler = preprocessing.RobustScaler() #scaling with outliers
#scaler = preprocessing.QuantileTransformer() #non-linear transformation
#scaler = preprocessing. Normalizer()
nov_processed = pd.DataFrame(scaler.fit_transform(nov_processed), columns=nov_processed.columns, index=nov_processed.index) 

nov_processed.to_hdf('data.hdf5', 'Datataset1/X')


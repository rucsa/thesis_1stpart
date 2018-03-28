from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

#X = pd.read_hdf('data.hdf5', 'Datataset1/X')
X = pd.read_hdf('dataWithOUTliers.hdf5', 'Datataset1/X')
y = X.loc[:,['analyst rating']]

#X.corr()
writer = pd.ExcelWriter('../resultscorr.xlsx', engine='xlsxwriter')
corr = X[X.columns].apply(lambda x: x.corr(X['analyst rating']))
corr.to_excel(writer, sheet_name='Sheet1')



writer = pd.ExcelWriter('../resultslinreg.xlsx', engine='xlsxwriter')

''' plot linear regression trendline '''
''' individual feature correlation - without outliers'''
output = pd.DataFrame(columns=['Feature', 'R_squared', 'Correlation'])
index = 0
features = X.columns.values.tolist()
features.remove('analyst rating')

for item in features:
    p = X[[item]].dropna().values
    #min = np.percentile(p, 25)
    max = np.percentile(p, 75)
    #print '%s values should be between %s and %s' % (item, 0, max)
    pe = X[[item, 'analyst rating']]
    pe = pe[np.isfinite(pe[item])]
    #pe = pe.loc[pe[item] <= max]
    anr = pe[['analyst rating']].values
    pe = pe[[item]].values
    fig = plt.figure()
    plt.ylabel('Analyst rating')
    plt.xlabel(item)
    # linregress needs one dimentional array - obtained through slicing
    plt.scatter(pe, anr, s=7)
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(pe[:,0], anr[:,0])
    plt.plot(pe, intercept + slope*pe, 'r', label='fitted line')
    plt.show()
    nl = '\n'
    output.loc[index] = ({"Feature": item, "R_squared": rvalue**2, "Correlation": rvalue})
    index = index + 1

output.to_excel(writer, sheet_name='Sheet1')
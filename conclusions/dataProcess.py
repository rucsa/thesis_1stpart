import scipy as sy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import utils as ut
from scipy import stats


X = pd.read_hdf('dataWithOUTliers.hdf5', 'Datataset1/X')
np.random.seed(47)

#feat_labels = ['total assets', 'quick ratio', 'gross profit', 'operational cash flow', 'market cap', 
#              'volatility 30 days', 'return last year','PSR']
target = pd.DataFrame(X, columns=['analyst rating'])
X = X.drop(['analyst rating'], axis=1)
#X = X[feat_labels]
#X = X.drop(['Returns 2016'], axis=1)
X = sm.add_constant(X)

targetFlat = target.values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, targetFlat,
                                                    test_size = 0.3)                                               
model = sm.OLS(y_train, X_train).fit() 
predictions = model.predict(X_test)
pe = X.values
plt.scatter(predictions, y_test, s=30, c='r', marker='.')

plt.yticks(np.arange(min(y_test), max(y_test), 1))
plt.xlabel("Independent variables")
plt.ylabel('Dependent variable: analyst rating')
plt.show()
#print('Dependent variable: analyst rating {}'.format(feat_labels))
print ("MSE: {}".format(model.mse_model))
print (model.summary())


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"
f = open('myreg.tex', 'w')
f.write(beginningtex)
f.write(model.summary().as_latex())
f.write(endtex)
f.close()
plt.close()
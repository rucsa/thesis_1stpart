from dataProcessing import X, y, features
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import utils as util
import pandas as pd

#feature_names=['Feature %d' % i for i in X]

#data = pd.read_hdf('data.hdf5', 'Datataset1/X')
X = pd.DataFrame(X)
y = pd.DataFrame(y)
#colors=util.generate_n_colors(len((X.columns), pastel_factor = 0.9)
#colors = {col:colors[i] for i, col in enumerate(X.columns)}
colors=[]
for i in range(0, X.shape[1]):
    colors.append(util.generate_new_color(colors, pastel_factor = 0.9))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state=0)

fig = plt.figure()
ax = plt.subplot(111)
n_alpha = 200
alphas = np.logspace(-10, -2, n_alpha)
coefs =[]

for a in alphas:
    lr = LassoCV(cv=4, alpha=a, fit_intercept=False, random_state=0)
    lr.fit(X, y)
    coefs.append(lr.coef_)
    

#weights=pd.Series(1, index=feature_names)

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.axis('tight')
plt.title('Lasso coefficients as a function of the regularization')

#for column, color in zip(range(weights.shape[0]), colors):
#    plt.plot(params, weights[:, column],
#             label=X.columns[column+1],
#             color=color)
#    plt.axhline(0, color='black', linestyle='--', linewidth=3)
#    plt.xlim([10**(-5), 10**5])
#    plt.ylabel('weight coefficient')
#    plt.xlabel('C')
#    plt.xscale('log')
#    plt.legend(loc='upper left')
#    ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03),
#              ncol=1, fancybox=True)
#    plt.show()
    
#    
#for colname, col in X.columns.items():
#    colors[colname]
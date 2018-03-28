from sklearn.linear_model import LassoLarsCV, LassoCV, Lasso, lasso_path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut


X = pd.read_hdf('dataWithOUTliers.hdf5', 'Datataset1/X')

X0 = X[['adjusted beta', 'volatility 360 days', 'return last year', 'market cap', 'net income', 'analyst rating']]
X1 = X[['return last year', 'quick ratio', 'PSR', 'market cap', 'adjusted beta', 'returns last 6 months', 
             'volatility 360 days', 'size', 'volatility 30 days', 'volatility 90 days', 'return last 3 month', 
             'P/E', 'EPS', 'analyst rating']]
X2 = X[['net income', 'total assets', 'volatility 90 days', 'gross profit', 'operational cash flow', 
              'volatility 30 days', 'PSR', 'quick ratio', 'returns last 5 years', 'return last year', 'analyst rating']]
X3 = X[['total assets', 'quick ratio', 'gross profit', 'operational cash flow', 'market cap', 
              'volatility 30 days', 'return last year','PSR', 'analyst rating']]


X = X3
y = X[['analyst rating']].values.flatten()
X = X.drop(['analyst rating'], axis=1)

np.random.seed(47)
#X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                    test_size = 0.3)
#ut.send_to_hd(X_train, X_test, y_train, y_test)

cv=list(KFold(n_splits=3, random_state=10).split(X,y))
cv=[list(cv)[0]]

#
#model = LassoCV(cv=10, precompute=False, fit_intercept=True).fit(X_train, y_train)
model = LassoCV(cv=cv, precompute=False, fit_intercept=True).fit(X, y)
results = dict(zip(X.columns, model.coef_))
print (results)

train_idx=cv[0][0]
test_idx=cv[0][1]

X_train = X.iloc[train_idx[0]:train_idx[train_idx.size-1]]
X_test = X.iloc[test_idx[0]:test_idx[test_idx.size-1]]
train_idx = train_idx[:-1]
test_idx = test_idx[:-1]
y_train = np.array([y[i] for i in train_idx])
y_test = np.array([y[i] for i in test_idx])

ut.send_to_hd(X_train, X_test, y_train, y_test)

# MSE
m_alpha = -np.log10(model.alphas_)
plt.figure(figsize=(8,8))
lass = model.mse_path_
plt.plot(m_alpha, lass, ':')
plt.plot(m_alpha, model.mse_path_, 'k', label='Average across folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
#plt.title('Mean squared error for each fold')
plt.show()
print('alpha is {}'.format(model.alpha_))

# coefficient prograssion
colors = []

features = X.columns.values.tolist()
plt.figure(figsize=(6,6))
eps = 5e-3
path = lasso_path(X, y, eps, fit_intercept=False)
neg_log_alphas_lasso = -np.log10(path[0])

for i, f in zip(path[1], features):
    c = ut.generate_new_color(colors, 0)
    plt.plot(neg_log_alphas_lasso, i.T, label=f, c=c, linewidth=1.5)
#plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha')
plt.ylabel('coefficients')
plt.xlabel('-log(alpha)')
#plt.title('Regularization Path for Lasso')
plt.legend(loc=9, bbox_to_anchor=(1.23, 1))
#plt.show()

train_error = mean_squared_error(y_train, model.predict(X_train))
test_error = mean_squared_error(y_test, model.predict(X_test))

# MSE
print ('training error {}'.format(train_error))
print ('test error {}'.format(test_error))
# R squared
print ('R_sqr for training {}'.format(model.score(X_train, y_train)))
print ('R_sqr for test {}'.format(model.score(X_test, y_test)))




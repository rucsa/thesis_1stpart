from dataProcessing import y
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = pd.read_hdf('data.hdf5', 'Datataset1/X')
y = pd.DataFrame(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state=0)

model = LassoLarsCV(cv=10, precompute=False, fit_intercept=False).fit(X_train, y_train)
results = dict(zip(X.columns, model.coef_))
print (results)

# MSE
m_alpha = -np.log10(model.cv_alphas_)
plt.figure()
lass = model.mse_path_
plt.plot(m_alpha, lass, ':')
plt.plot(m_alpha, model.mse_path_.mean(axis=-1), 'k', label='Average across folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('MSE for each fold')
plt.show()

# coefficient prograssion
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha')
plt.ylabel('coefficients')
plt.legend()
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Pregression for Lasso Lars')
plt.show()

train_error = mean_squared_error(y_train, model.predict(X_train))
test_error = mean_squared_error(y_test, model.predict(X_test))

# MSE
print ('training error {}'.format(train_error))
print ('test error {}'.format(test_error))
# R squared
print ('R_sqr for training {}'.format(model.score(X_train, y_train)))
print ('R_sqr for test {}'.format(model.score(X_test, y_test)))
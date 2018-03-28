from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataProcessing import y
y = pd.DataFrame(y).values.ravel()


#a=1
#b=2
#
#x=np.linspace(0,10)
#y=a*x+b+0.5*np.random.randn(len(x))
#
#
#x = x.reshape(-1,1)
#y=y.reshape(-1,1)

#fit on this data, you should recover a=1, b=2

x = pd.read_hdf('data.hdf5', 'Datataset1/X')

delete_list = ['volatility 30 days', 'volatility 360 days', 'P/E', 'returns last 5 years', 
               'Sector', 'region', 'PSR'] # lasso feature selection


x = x.drop(delete_list, axis=1)

model = LinearRegression(fit_intercept=True)
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.5,
                                                    random_state=0)



model.fit(X_train, y_train)
predictions = model.predict(X_test)

#plt.plot(x,y, label='data')
plt.plot(X_test, predictions, label='linear prediction')


model = LassoLars(precompute=False, alpha=0,
                                      fit_intercept=True)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.plot(X_test, predictions, label='lasso prediction')

plt.legend()
plt.show()

models = []
models.append(('LinearRegression', LinearRegression(fit_intercept=True)))
models.append(('LassoLars', LassoLars(precompute=False, alpha=0,
                                      fit_intercept=True)))

results = []
for name, model in models:
    scores = cross_validate(model, x, y, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
    
    RSS_train = -len(y)*scores['train_neg_mean_squared_error']
    MSE_train = -np.mean(scores['train_neg_mean_squared_error'])
    r2_train = np.mean(scores['train_r2'])
    
    RSS_test = -len(y)*scores['test_neg_mean_squared_error']
    MSE_test = -np.mean(scores['test_neg_mean_squared_error'])
    r2_test = np.mean(scores['test_r2'])
    
    results.append({"model":name, 
                    "RSS train":np.mean(RSS_train), "RSS test":np.mean(RSS_test), 
                    "MSE train":MSE_train, "MSE test":MSE_test, 
                    "R_squared train":r2_train, "R_squared test":r2_test})
results = pd.DataFrame(results)
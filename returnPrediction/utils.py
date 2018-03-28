import scipy as sy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def MLR (data, targetF):
    target = pd.DataFrame(data, columns=[targetF])
    data = data.drop([targetF], axis=1)
    data = sm.add_constant(data)
    
    np.random.seed(47)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size = 0.3)
    target = target.values.flatten()
    model = sm.OLS(y_train, X_train).fit() 
    predictions = model.predict(X_test)
    
    plt.scatter(predictions, y_test, s=30, c='r', marker='+', zorder=10)
    plt.yticks(np.arange(min(target), max(target), 1.0))
    plt.xlabel("Predicted Values from model")
    plt.ylabel("Actual {}".format(targetF))
    plt.show()
    print ("MSE: {}".format(model.mse_model))
    print (model.summary())

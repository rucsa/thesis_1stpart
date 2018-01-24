import time
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def count_duplicates(counted, dictionary):
    if (counted == len(dictionary)):
        print ('no duplicate keys in dict')
    else:
        print ('{} elements have the same key'.format(len(dictionary) - counted))
        
        
def processSubset(X, y, feature_set):
    # Fit model on feature_set and calculate RSS
    X_train, X_test, y_train, y_test = train_test_split(X[list(feature_set)], y)
    model = LinearRegression()
    regr = model.fit(X_train, y_train)
    predicted = regr.predict(X_test)
    mean_obs = np.mean(y_test)
    ss_res = np.sum((y_test - predicted)**2)
    ss_tot = np.sum((y_test - mean_obs)**2)
    r = 1-(ss_res/ss_tot)
    return {"model":regr, "RSS":ss_res, "R_sqr":r}

def getBest(X, y, k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(X, y, combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models["RSS"].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    return best_model


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

import utils as util
import dataProcessing as dp
from dataProcessing import nov_processed
from itertools import combinations

import time

features = list(nov_processed.columns.values)
results = []

nov_processed = nov_processed.dropna(axis=0, how='any')

y = nov_processed.loc[:, 'analyst rating']
X = nov_processed.drop(['analyst rating'], axis=1)

models_fwd = pd.DataFrame(columns=['model', 'RSS', 'R_squared', 'features'])
tic = time.time()
predictors = []
model = LinearRegression()

def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    model = LinearRegression()
    regr = model.fit(y,X[list(feature_set)])
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def backward(predictors):
    tic = time.time()
    results = []

    for combo in combinations(predictors, len(predictors)-1):
        results.append(processSubset(combo))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]

    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)-1, "predictors in", (toc-tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model

models_bwd = pd.DataFrame(columns=["RSS", "model"], index = range(1,len(X.columns)))

tic = time.time()
predictors = X.columns

while(len(predictors) > 1):
    models_bwd.loc[len(predictors)-1] = backward(predictors)
    predictors = models_bwd.loc[len(predictors)-1]["model"].model.exog_names

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")



import time
import itertools
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib.colors import ListedColormap
import matplotlib as plt
import random


def send_to_hd (X_train, X_test, y_train, y_test):
    X_train.to_hdf('X_train.hdf5', 'Datataset1/X')
    X_test.to_hdf('X_test.hdf5', 'Datataset1/X')
    y_train = pd.DataFrame(y_train)
    y_train.to_hdf('y_train.hdf5', 'Datataset1/X')
    y_test = pd.DataFrame(y_test)
    y_test.to_hdf('y_test.hdf5', 'Datataset1/X')
    return True

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = max([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def scaleOutMarketCap (df, features):
    for feature in features:
        df.loc[:,feature] = df.loc[:,feature]/df.loc[:,'market cap'] * 10000
    return df

def categoricalToNumeric (df, encoder):
    for feature in (df.columns.values):
        df.loc[:,feature] = encoder.fit_transform(df.loc[:,feature].values)
    return df

def encodeMarketCap(df):
    mc = list((df.loc[:,'market cap'] * 1000000).values)
    for i in range (0, len(mc)):
        if (10000000000 <= mc[i] ):
            mc[i] = 4 #'large cap'
        elif (2000000000 <= mc[i] and mc[i] < 10000000000):
            mc[i] = 3 #'mid cap'
        elif (300000000 <= mc[i] and mc[i] < 2000000000):
            mc[i] = 2 #'small cap'
        elif (50000000 <= mc[i] and mc[i] < 300000000):
            mc[i] = 1 #'micro cap'
        elif (mc[i] < 50000000):
            mc[i] = 0 #'nano cap'
    return pd.DataFrame(mc, columns=['size'])
    

def count_duplicates(counted, dictionary):
    if (counted == len(dictionary)):
        print ('no duplicate keys in dict')
    else:
        print ('{} elements have the same key'.format(len(dictionary) - counted))
        
def forward(X, y, predictors, model):

    remaining_predictors = [p for p in X.columns if p not in predictors]
    tic = time.time()
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(X, y, predictors+[p], model))
    
    models = pd.DataFrame(results)
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    
    return best_model
        
def processSubset(X, y, feature_set, model):
    # Fit model on feature_set and calculate RSS
    X = X[list(feature_set)]
    n = len(y)
    scores = cross_validate(model, X, y, cv=10, scoring=('r2', 'neg_mean_squared_error'))
    RSS = -n*scores['test_neg_mean_squared_error']
    assert np.all(scores['test_r2'] <=1), "Invalid R2"
    r2 = np.mean(scores['test_r2'])
    return {"model":model, "RSS":np.mean(RSS), "R_squared":r2, "features": feature_set}


def getBest(X, y, k, model):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(X, y, combo, model))
    models = pd.DataFrame(results)
    best_model = models.loc[models["RSS"].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    return best_model


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'grey', 'cyan', 'black')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X.loc[:, 'returns last 5 years'].min() - 1, X.loc[:, 'returns last 5 years'].max() + 1
    x2_min, x2_max = X.loc[:, 'PSR'].min() - 1, X.loc[:, 'PSR'].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourof(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha = 0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

def featuresCorrWithoutOutliers(features, mainDataFrame, model):
    count = 0
    results = {}
    
    for i in range (1, 26):
        print ('computing {} features'.format(i))
        for feature_set in combinations(features, i):
            feature_set = list(feature_set)
            feature_set.append('analyst rating')
            X = mainDataFrame.loc[:, feature_set].dropna()
            y = X.loc[:, 'analyst rating']
            X = X.drop(['analyst rating'], axis=1)
    
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
    
            residuals = y_test - predicted
            mean_obs = np.mean(y_test)
            ss_res = np.sum((y_test - predicted)**2)
            ss_tot = np.sum((y_test - mean_obs)**2)
            mse = np.mean((predicted - y_test)**2)
            r = model.score(X_test, y_test)
            #print ('Mean squared error = {}'.format(mse))
            #print ('lreg R_squared is {} | Calculated R_squared is {}\n'.format(r, 1-(ss_res/ss_tot)))
            feature_set.remove('analyst rating')
            key = [feature_set]
            results[r] = key
            count = count + 1
        print ('finished computing {} features'.format(i))
    return results

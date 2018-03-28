from dataProcessing import y_class
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_hdf('data.hdf5', 'Datataset1/X')
y_class = pd.DataFrame(y_class).values.ravel()
a, b, c, d, e = 0, 0, 0, 0, 0
for y in y_class:
    if (y==1): 
        a=a+1
    elif (y==2): 
        b=b+1
    elif (y==3): 
        c=c+1
    elif (y==4):
        d=d+1
    else: 
        e=e+1
    
print ("target contains /n {} strong sell /n {} sell /n {} hold /n {} buy /n {} strong buy".format(a, b, c, d, e))
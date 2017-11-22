import pandas as pd
import numpy as np
from sklearn import preprocessing

data2 = pd.read_csv("tutorial_data/portfolio_data.csv", sep=';', header = 1);

#data2 = data2.applymap(str);
data2 = data2.replace({'%':''}, regex=True);
#print (data2);

data_scaled = preprocessing.scale(data2)
print (data_scaled);


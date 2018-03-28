from dataProcessing import y_class
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LassoLars
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_hdf('data.hdf5', 'Datataset1/X')
y = pd.DataFrame(y_class).values.ravel()
#a, b, c, d, e = 0, 0, 0, 0, 0
#for i in y:
#    if (i==1): 
#        a=a+1
#    elif (i==2): 
#        b=b+1
#    elif (i==3): 
#        c=c+1
#    elif (i==4):
#        d=d+1
#    else: 
#        e=e+1
#    
#print ("target contains /n {} strong sell /n {} sell /n {} hold /n {} buy /n {} strong buy".format(a, b, c, d, e))

# shuffle rows

X = X.sample(frac=1).reset_index(drop=True)

# split samples
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state=0)
# fit
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=45, loss='deviance')

model2 = MLPClassifier(hidden_layer_sizes=(100, ), activation="logistic", solver="adam", alpha=0.0001, batch_size="auto", 
                       learning_rate="adaptive", learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, 
                       random_state=27, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model3 = RandomForestClassifier(bootstrap=True, criterion='gini', 
                                                         max_depth=2, 
                                                         max_features='auto', 
                                                         max_leaf_nodes=None,
                                                         min_impurity_decrease=0.0, 
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1, 
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0, 
                                                         n_estimators=10, n_jobs=1,
                                                         oob_score=False, random_state=0, 
                                                         verbose=0, warm_start=False)

prediction = model3.fit(X_train, y_train).predict(X_test)

m = confusion_matrix(y_test, prediction)
p = precision_score(y_test, prediction, average='micro')
r = recall_score(y_test, prediction, average='micro')
f = f1_score(y_test, prediction, average='micro')
k = cohen_kappa_score(y_test, prediction)

print("Precision {}".format(p))
print("Recall {}".format(k))
print("F1 {}".format(f))


print("Kappa {}".format(k))
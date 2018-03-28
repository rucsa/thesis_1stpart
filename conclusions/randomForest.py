from dataProcessing import y
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#X = pd.read_hdf('data.hdf5', 'Datataset1/X')
X = pd.read_hdf('dataWithOUTliers.hdf5', 'Datataset1/X')

y = X.loc[:,['analyst rating']]
X = X.drop(['analyst rating'], axis=1)

feat_labels = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state=0)

# it's not necessary for data to be standardized or normalized 
model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=1,
           oob_score=False, random_state=1, verbose=0, warm_start=False)
model.fit(X_train, y_train)
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range (X_train.shape[1]):
    print ('{}) {} {}'.format(f+1, feat_labels[indices[f]], 
          importances[indices[f]]))

plt.title('Feature importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


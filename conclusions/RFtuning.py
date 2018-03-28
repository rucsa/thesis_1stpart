import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut

from lasso_cross_val import X, y

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV



rfr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5, 
                              max_features='auto', 
                              max_leaf_nodes=None, 
                              min_impurity_decrease=0.0, 
                              min_impurity_split=None, 
                              min_samples_leaf=50, 
                              min_samples_split=2, 
                              min_weight_fraction_leaf=0.0, 
                              n_estimators=100, 
                              n_jobs=-1, 
                              oob_score=True, 
                              random_state=0, 
                              verbose=0, 
                              warm_start=False)
gbr =  GradientBoostingRegressor(loss='ls', 
                                 learning_rate=0.06, 
                                 n_estimators=100, 
                                 subsample=0.6, 
                                 criterion='friedman_mse', 
                                 min_samples_split=3,
                                 min_samples_leaf=3, 
                                 min_weight_fraction_leaf=0.0, 
                                 max_depth=3, 
                                 min_impurity_decrease=0.0, 
                                 min_impurity_split=None, 
                                 init=None, 
                                 random_state=None,
                                 max_features=12, 
                                 alpha=0.04, 
                                 verbose=0, 
                                 max_leaf_nodes=None, 
                                 warm_start=False, 
                                 presort='auto')
max_depth_options = [2, 3, 5, 10, 15, 20, 50]
min_sample_split_options = [2, 3, 5, 10, 15, 20, 50]
min_sample_leaf_options = [1, 2, 3, 4, 5, 10, 20, 50]
max_features_options = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
subsample_options = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
learning_rate_options = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 1.8, 2, 3, 4, 5, 6]


param_grid = [
        {
                #'max_depth':max_depth_options, #3
                #'min_samples_leaf': min_sample_leaf_options, #3
                #'min_samples_split': min_sample_split_options, #3
                #'max_depth': max_depth_options, #3
                #'max_features': max_features_options, #12
                #'subsample': subsample_options, #0.6
                'learning_rate': learning_rate_options, #0.04
        }
        ]


grid = GridSearchCV(gbr, param_grid=param_grid, scoring=('r2', 'neg_mean_squared_error'), fit_params=None, n_jobs=1, 
             iid=True, refit='neg_mean_squared_error', cv=4, verbose=0, pre_dispatch='2*n_jobs', 
             error_score='raise', return_train_score=False)
train = grid.fit (X, y)
grid_results = pd.DataFrame(grid.cv_results_)
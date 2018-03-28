# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:10:45 2018

@author: RuxandraV
"""


import numpy as np

import statsmodels.api as sm

spector_data = sm.datasets.spector.load()

spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

# Fit and summarize OLS model
mod = sm.OLS(spector_data.endog, spector_data.exog)

res = mod.fit()

print(res.summary())
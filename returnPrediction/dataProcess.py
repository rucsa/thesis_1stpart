import scipy as sy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import utils as ut

'''' upload data '''
data = pd.read_excel('../../Data/OnlyValues_Returns_prediction.xlsx')
data = data[['PE', 'Volatility360', 'ANR', 'Returns 2016', 'Returns 2017']]

data.loc[:,'PE'] = data.loc[:,'PE']/100
data.loc[:,'Volatility360'] = data.loc[:,'Volatility360']/100

X = data.copy()
np.random.seed(47)

target = pd.DataFrame(X, columns=['ANR'])
X = X.drop(['ANR'], axis=1)
X = X.drop(['Returns 2017'], axis=1)
#X = X.drop(['Returns 2016'], axis=1)
X = sm.add_constant(X)

targetFlat = target.values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, targetFlat,
                                                    test_size = 0.3)                                               
model = sm.OLS(y_train, X_train).fit() 
predictions = model.predict(X_test)

plt.scatter(predictions, y_test, s=30, c='r', marker='.')
plt.yticks(np.arange(min(y_test), max(y_test), 1))
plt.xlabel("Independent variables: P/E, Volatility 360 days, Returns 2016, intercept")
plt.ylabel('Actual ANR')
plt.tight_layout()
plt.show()
print ("MSE: {}".format(model.mse_model))
print (model.summary())
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"
f = open('myreg1.tex', 'w')
f.write(beginningtex)
f.write(model.summary().as_latex())
f.write(endtex)
f.close()
plt.close()


X = data.copy()
target = pd.DataFrame(X, columns=['Returns 2017'])
X = X.drop(['ANR'], axis=1)
X = X.drop(['Returns 2017'], axis=1)
X = sm.add_constant(X)

targetFlat = target.values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, targetFlat,
                                                    test_size = 0.3)   
model = sm.OLS(y_train, X_train).fit() 
predictions = model.predict(X_test)

plt.scatter(predictions, y_test, s=30, c='r', marker='.')
plt.yticks(np.arange(min(y_test), max(y_test), 0.5))
plt.xlabel("Independent variables: P/E, Volatility 360 days, Returns 2016, intercept")
plt.ylabel('Actual Returns 2017')
plt.tight_layout()
plt.show()
print ("MSE: {}".format(model.mse_model))
print (model.summary())
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"
f = open('myreg2.tex', 'w')
f.write(beginningtex)
f.write(model.summary().as_latex())
f.write(endtex)
f.close()
plt.close()


X = data.copy()
target = pd.DataFrame(X, columns=['Returns 2017'])
X = X.drop(['Returns 2016'], axis=1)
X = X.drop(['Returns 2017'], axis=1)
X = X.drop(['Volatility360'], axis=1)
X = X.drop(['PE'], axis=1)
X = sm.add_constant(X)

targetFlat = target.values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, targetFlat,
                                                    test_size = 0.3)  
model = sm.OLS(y_train, X_train).fit() 
predictions = model.predict(X_test)

plt.scatter(predictions, y_test, s=30, c='r', marker='.')
plt.yticks(np.arange(min(y_test), max(y_test), 0.5))
plt.xlabel("Independent variables: ANR, intercept")
plt.ylabel('Actual Returns 2017')
plt.tight_layout()
plt.show()
print ("MSE: {}".format(model.mse_model))
print (model.summary())
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"
f = open('myreg3.tex', 'w')
f.write(beginningtex)
f.write(model.summary().as_latex())
f.write(endtex)
f.close()
plt.close()
plt.close('all')
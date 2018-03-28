from dataProcessing import y
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

X = pd.read_hdf('data.hdf5', 'Datataset1/X')
y = pd.DataFrame(y)
X = pd.concat([X, y], axis=1)
feat_labels = X.columns
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
for i in range (1, X.shape[1]):
    feature = X.ix[:, i]
    bins = np.linspace(feature.min(), feature.max(), 100)
    plt.hist(feature, bins=bins, fc='#AAAAFF', normed=True)
    plt.title('Histogram of {}'.format(feature.name))
    file_name = 'histograms_simple/hist_' + str(i) + '.png'
    plt.savefig(file_name, format='png')

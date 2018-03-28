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
    feature = X.iloc[:, i]
    bins = np.linspace(feature.min(), feature.max(), 100)
    fig, ax = plt.subplots(2, 1, figsize=(5,5))
    plt.suptitle('{} Histogram and Gaussian Kernel Density'.format(feature.name))
    
    ax[0].hist(feature, bins=bins, fc='#AAAAFF', normed=True)
    file_name = 'histograms/hist_' + str(i) + '.png'
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(feature.values.reshape(-1,1))
    log_dens = kde.score_samples(X_plot)
    ax[1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')

    
    plt.savefig(file_name, format='png')
    plt.close(fig)
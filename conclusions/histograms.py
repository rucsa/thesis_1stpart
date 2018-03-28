from dataProcessing import y
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

#X = pd.read_hdf('data.hdf5', 'Datataset1/X')
X = pd.read_hdf('dataWithOUTliers.hdf5', 'Datataset1/X')
y = X.loc[:,['analyst rating']]
X = X.drop(['analyst rating'], axis=1)

feat_labels = ['adjusted beta', 'volatility 360 days', 'return last year', 'market cap', 'net income']
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

for i in feat_labels: # range (1, X.shape[1]):
    feature = X.ix[:, i]
    bins = np.linspace(feature.min(), feature.max(), 100)
    fig, ax = plt.subplots(2, 1, figsize=(4,4))
    #plt.title('Histogram of {}'.format(feature.name))
    ax[0].hist(feature, bins=bins, color='#87CEFA', normed=True)
    file_name = 'histograms/hist_' + str(i) + '.png'
    plt.ylabel('Frequency')
    plt.xlabel('Values of {}'.format(feature.name))
    plt.tight_layout()
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(feature.values.reshape(-1,1))
    log_dens = kde.score_samples(X_plot)
    ax[1].fill(X_plot[:, 0], np.exp(log_dens), color='#87CEFA')
    #("Gaussian Kernel Density")
    
    plt.savefig(file_name, format='png')
plt.close(fig)
#Adapted from https://github.com/SheffieldML/GPclust

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
from scipy import stats
import sys
print('\n'.join(sys.path))
import GPy
import GPclust

#Input the data and specify the names of the output files
data = '.csv' #Insert Z-transformed timeseries data for clustering, in csv format, reps should be sequential
fig_output_name = 'clusters_fig.png'
heatmap_output_name = 'heatmap_fig.png'
data_output_name = 'clusters_probabilities.csv'

#Prepare the data for clustering
df = pd.read_csv(data, header = 0)
df = df.drop(columns='FBgn') #Remove IDs if data contains them
Y = df.to_numpy()
#Defining array of timepoints
timepoints = ([[95], [105], [115], [125], [145], [160], [175], [190], [205], [220],[95], [105], [115], [125], [145], [160], [175], [190], [205], [220], [95], [105], [115], [125], [145], [160], [175], [190], [205], [220]])
X = np.array(timepoints)

# Clustering
k_underlying = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=50.)
k_corruption = GPy.kern.Matern52(input_dim=1, variance=0.1, lengthscale=50.) + GPy.kern.White(1, variance=0.05)
m = GPclust.MOHGP(X, k_underlying, k_corruption, Y, K=10, prior_Z='DP', alpha=1.0)
m.hyperparam_opt_interval = 1000 # how often to optimize the hyperparameters
m.hyperparam_opt_args['messages'] = False # turn off the printing of the optimization
m.optimize()
m.systematic_splits(verbose=False)

# Plotting and outputting cluster assignments
m.reorder() # plot the biggest clusters at the top
plt.figure(figsize=(18,18))
m.plot(on_subplots=True, colour=True, in_a_row=False, newfig=False, min_in_cluster=1, joined=False, ylim=(-2,2))
plt.savefig(fig_output_name)

phi = m.phi
_=plt.matshow(phi.T, cmap=plt.cm.hot, vmin=0, vmax=1, aspect='auto')
plt.xlabel("Data index")
plt.ylabel("Cluster index")
_=plt.colorbar()
#heatmap of posterior assignment probabilities
plt.savefig(heatmap_output_name)

#output of matrix phi (N x K) where each element (ΦNK) represents the probability that the Nth transcript is assigned to the Kth cluster
np.savetxt(data_output_name, phi, delimiter=",")

##########################################################################################
# Copute correlation between pre-mRNA and mRNA
##########################################################################################

import gpflow
import numpy as np
import matplotlib
import pandas as pd
from gpflow.utilities import print_summary
from utilities import fit_rbf, fit_noise
import matplotlib.pyplot as plt# The lines below are specific to the notebook format
import sys
sys.path.append('..')
from utils.utilities import init_hyperparameters
from scipy import stats


np.random.seed(100)

# Load the data
data1 = pd.read_csv('data/LB_GP_TS.csv', sep=",")
data2 = pd.read_csv('data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")

t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
rep_no = 3
t = np.hstack((t0,t0,t0))[:,None]

names_transcripts =  pd.read_csv('data/LB_zyg95_clusters_genes.csv', sep=",")
names_transcripts.columns = ['gene', 'target_id', 'cluster']

genesID = []
trID = []

corr = []
p_val = []
cluster_list = []
cluster_int_list = []

for i in range(0,590):
    try:
        gene_id = names_transcripts['gene'].iloc[i]
        tr_id = names_transcripts['target_id'].iloc[i]
        cluster = names_transcripts['cluster'].iloc[i]

        data11 = data1[data1['FBtr'] == tr_id]
        data22 = data2[data2['FBgn'] == gene_id]

        data_m = data11[data11['FBtr'] == tr_id].iloc[0][1:31].to_numpy()
        data_p = data22[data22['FBgn'] == gene_id].iloc[0][31:61].to_numpy()

        Y_m = data_m.astype(np.float64)
        Y_p = data_p.astype(np.float64)

        # Y and Z are numpy arrays or lists of variables
        correlation = stats.pearsonr(Y_m, Y_p)
        corr.append(correlation[0])
        p_val.append(correlation[1])
        genesID.append(str(gene_id))
        trID.append(str(tr_id))
        cluster_list.append(str(cluster))

    except:
        print('data werent found')

corr = np.asarray(corr)
p_val = np.asarray(p_val)
df_genesID = pd.DataFrame(genesID)
df_trID = pd.DataFrame(trID)

df = pd.concat([df_genesID, df_trID, pd.DataFrame(corr), pd.DataFrame(p_val)], axis=1)

df.columns = ['genesID','trID', 'corr', 'p_val']
df.to_csv('../output/correlation_LB_zyg95_clusters_genes.csv',index=False)

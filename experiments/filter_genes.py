'''
In this file we filter out the genes that do not have dynamics;
1) Fit individual GPs for mRNA and prem_RNA with noise kernel;
2) Fir individual GPs for mRNA and prem_RNA with RBF kernels;
3) Do likelihood ratio test;
4) Outputs file with p-values for loglikelihood ratio test; p_val<alpha => gene is expressed;
'''

import gpflow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from gpflow.utilities import print_summary
import sys
sys.path.append('..')
from utils.utilities import fit_rbf,fit_noise, fit_noise_rbf, init_hyperparameters
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2


np.random.seed(100)

data1 = pd.read_csv('../data/LB_GP_TS.csv', sep=",")
data2 = pd.read_csv('../data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")
names_transcripts =  pd.read_csv('../data/zygotic_tr_95_genes.csv', sep=";")
names_transcripts = names_transcripts.dropna(axis='rows')

t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
rep_no = 3
t = np.hstack((t0,t0,t0))[:,None]

mlik_noise_mrna = []
mlik_noise_premrna = []
mlik_rbf_mrna = []
mlik_rbf_premrna = []

lengthscale_rbf_mrna = []
variance_rbf_mrna = []

lengthscale_rbf_premrna = []
variance_rbf_premrna = []

variance_noise_mrna = []
variance_noise_premrna = []

genesID = []
trID = []

for i in range(len(names_transcripts)-1):
#for i in range(0,5):

    gene_id = names_transcripts['FBgn'].iloc[i]
    tr_id = names_transcripts['FBtr'].iloc[i]

    data11 = data1[data1['FBtr'] == tr_id]
    data22 = data2[data2['FBgn'] == gene_id]

    data_m = data11[data11['FBtr'] == tr_id].iloc[0][1:31].to_numpy()
    data_p = data22[data22['FBgn'] == gene_id].iloc[0][31:61].to_numpy()

    Y_m = data_m.astype(np.float64)
    Y_p = data_p.astype(np.float64)

    data_full = (t.reshape(-1,1), Y_m.reshape(-1,1))
    genesID.append(str(gene_id))
    trID.append(str(tr_id))

    '''
    Fit RBF mRNA
    '''
    count_Cholesky_fail = 0  # number of trial to fix Cholesky decomposition
    fail = False
    while count_Cholesky_fail < 20:
        try:
            if fail:
                init_lengthscale, init_variance = init_hyperparameters(t0)
                fail = False
            m, lik = fit_rbf(data_full, init_lengthscale, init_variance)

            #dist_mean, dist_var = distance_from_mean(m, data_full, num_predict= 100)

            mlik_rbf_mrna.append(lik)
            lengthscale_rbf_mrna.append(np.asarray(m.kernel.lengthscales.read_value()))
            variance_rbf_mrna.append(np.asarray(m.kernel.variance.read_value()))

        except:
            fail = True
            count_Cholesky_fail = count_Cholesky_fail + 1
            continue
            #print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')
        break

    if(count_Cholesky_fail == 20):
         lengthscale_rbf_mrna.append(np.nan)
         variance_rbf_mrna.append(np.nan)
         mlik_rbf_mrna.append(np.nan)
         print('Error in RBF mRNA')
         print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')

    '''
    Fit noise mRNA
    '''
    count_Cholesky_fail = 0  # number of trial to fix Cholesky decomposition
    fail = False
    while count_Cholesky_fail < 20:
        try:
            if fail:
                init_lengthscale, init_variance = init_hyperparameters(t0)
                fail = False
            m, lik = fit_noise(data_full, init_variance)

            mlik_noise_mrna.append(lik)
            variance_noise_mrna.append(np.asarray(m.kernel.variance.read_value()))

        except:
            fail = True
            count_Cholesky_fail = count_Cholesky_fail + 1
            continue
            #print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')
        break

    if(count_Cholesky_fail == 20):
         variance_noise_mrna.append(np.nan)
         mlik_noise_mrna.append(np.nan)
         print('Error in noise mRNA')
         #print(data.FBgn.iloc[i])
         print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')

    '''
    Fit RBF premRNA
    '''
    data_full = (t.reshape(-1,1), Y_p.reshape(-1,1))

    count_Cholesky_fail = 0  # number of trial to fix Cholesky decomposition
    fail = False
    while count_Cholesky_fail < 20:
        try:
            if fail:
                init_lengthscale, init_variance = init_hyperparameters(t0)
                fail = False
            m, lik = fit_rbf(data_full, init_lengthscale, init_variance)

            #dist_mean, dist_var = distance_from_mean(m, data_full, num_predict= 100)

            mlik_rbf_premrna.append(lik)
            lengthscale_rbf_premrna.append(np.asarray(m.kernel.lengthscales.read_value()))
            variance_rbf_premrna.append(np.asarray(m.kernel.variance.read_value()))

        except:
            fail = True
            count_Cholesky_fail = count_Cholesky_fail + 1
            continue
            #print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')
        break

    if(count_Cholesky_fail == 20):
         #mlik_rbf_premrna.append(0.0)
         lengthscale_rbf_premrna.append(np.nan)
         variance_rbf_premrna.append(np.nan)
         mlik_rbf_premrna.append(np.nan)
         print('Error in RBF pre-mRNA')
         #print(data.FBgn.iloc[i])
         print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')

    '''
    Fit noise premRNA
    '''
    data_full = (t.reshape(-1,1), Y_p.reshape(-1,1))

    count_Cholesky_fail = 0  # number of trial to fix Cholesky decomposition
    fail = False
    while count_Cholesky_fail < 20:
        try:
            if fail:
                init_lengthscale, init_variance = init_hyperparameters(t0)
                fail = False
            m, lik = fit_noise(data_full, init_variance)
            mlik_noise_premrna.append(lik)
            variance_noise_premrna.append(np.asarray(m.kernel.variance.read_value()))

        except:
            fail = True
            count_Cholesky_fail = count_Cholesky_fail + 1
            continue
        break

    if(count_Cholesky_fail == 20):
         variance_noise_premrna.append(np.nan)
         mlik_noise_premrna.append(np.nan)
         print('Error in noise pre-mRNA')
         print('Can not fit a Gaussian process, Cholesky decomposition was not successful.')

df_genesID = pd.DataFrame(genesID)
df_trID = pd.DataFrame(trID)

loglik_ratio_mrna = -np.asarray(mlik_rbf_mrna) - (-np.asarray(mlik_noise_mrna))
loglik_ratio_premrna = -np.asarray(mlik_rbf_premrna) - (-np.asarray(mlik_noise_premrna))

lengthscale_rbf_mrna = np.asarray(lengthscale_rbf_mrna)
variance_rbf_mrna = np.asarray(variance_rbf_mrna)
lengthscale_rbf_mrna = np.asarray(lengthscale_rbf_premrna)
variance_rbf_mrna = np.asarray(variance_rbf_premrna)
variance_noise_mrna = np.asarray(variance_noise_mrna)
variance_noise_premrna = np.asarray(variance_noise_premrna)

df = pd.concat([df_genesID, df_trID, pd.DataFrame(mlik_rbf_mrna), pd.DataFrame(mlik_noise_mrna), pd.DataFrame(loglik_ratio_mrna),
                pd.DataFrame(mlik_rbf_premrna), pd.DataFrame(mlik_noise_premrna), pd.DataFrame(loglik_ratio_premrna),
                pd.DataFrame(lengthscale_rbf_mrna), pd.DataFrame(variance_rbf_mrna), pd.DataFrame(variance_noise_mrna),
                pd.DataFrame(lengthscale_rbf_premrna), pd.DataFrame(variance_rbf_premrna), pd.DataFrame(variance_noise_premrna)], axis=1)

df.columns = ['genesID','trID', 'rbf_mrna', 'noise_mrna', 'loglik_ratio_mrna', 'rbf_premrna', 'noise_premrna', 'loglik_ratio_premrna',
              'lengthscale_rbf_mrna', 'variance_rbf_mrna', 'variance_noise_mrna', 'lengthscale_rbf_premrna', 'variance_rbf_premrna', 'variance_noise_premrna']

df['p_val_mrna'] = chi2.sf(loglik_ratio_mrna ,1)
df['p_val_premrna'] = chi2.sf(loglik_ratio_premrna,1)

df.to_csv('../output/filtered_genes_zygotic_tr_95_genes.csv',index=False)

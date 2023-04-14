# GP_Transcription_Dynamics

Python implementation of the transcriptional regulation model with Gaussian processes using [GPflow](https://www.gpflow.org/) and [TensorFlow probability](https://www.tensorflow.org/probability).

## Dependencies

See requirenments.txt for full reproducibility; for lighter version of the code check simulations_python39 with lighter requirements. 

##  TRCD 

-- contains the transcriptional regulation model (custom implementation of GPR for stacked time series; transcriptional regulation kernel).

## Simulations:
-- simulated_examples.py runs an experiment on simulated data (generates the data, fits trcd model and runs MCMC).

## Simulations_python39:
-- Same as simulations, but with a few updates from latest version of the libraries (i.e., gpflow); 

-- Minimal requirements compared to the main directory; 

-- Runs on Mac M1, but tensorflow needs to be installed accordingly (see intructions for tensorflow installation [here](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706));

-- simulated_examples.py runs an experiment on simulated data (generates the data, fits trcd model and runs MCMC).

## Experiments:
Contains files for the experiments on real data. 

-- filter_genes.py/filter_genes_2rbf.py filtering of the genes (identifying differentially expressed genes);

-- fit_model_filtered_genes.py optimization of the parameters in transcriptional regulation model for genes that passed filtering;

-- mcmc_single_gene.py/mcmc_all_genes.py MCMC for uncertainty quantification using MALA on single gene/complete data set. 

## Utils:
Contains helper functions for loading the data/fitting the model/running MCMC. 

## Clustering:
-- source code for clustering time series with [GPclust](https://github.com/SheffieldML/GPclust).





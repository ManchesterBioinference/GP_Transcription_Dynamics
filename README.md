# GP_Transcription_Dynamics

Python implementation of the transcriptional regulation model with Gaussian processes using [GPflow](https://www.gpflow.org/) and [TensorFlow probability](https://www.tensorflow.org/probability).

## Dependencies

See requirenments.txt

##  TRCD 

-- contains the transcriptional regulation model (custom implementation of GPR for stacked time series; transcriptional regulation kernel).

## Simulations:
-- simulated_examples.py runs an experiment on simulated data (generates the data, fits trcd model and runs mcmc);

## Experiments:
Contains files for the experiments on real data. 

-- filter_genes.py filtering of the genes (identifying differentially expressed genes);

-- fit_model_filtered_genes.py optimization of the parameters in transcriptional regulation model;

-- mcmc_single_gene.py/mcmc_all_genes.py MCMC for uncertainty quantification using MALA. 

## Utils:
Contains helper functions for loading the data/fitting the model/running MCMC. 







# Sparse GEMINI

Preprint: [Sparse GEMINI for Joint Discriminative Clustering and Feature Selection](https://arxiv.org/abs/2302.03391)

Welcome to the Sparse GEMINI repo. This repo contains the official implementation of the sparse gemini algorithm as well as all the commands to reproduce the experiments.

## Model definition

If you are only interested in using the sparse gemini model, you can simply download the `*.py` files in the `sparse_gemini` folder.
The packages required to run the model are described in `requirements.txt`. Once installed, the command

```
python sparse_gemini/main.py -h
```

will give details on all parameters that can be used to toy with the model.

## Other folders

The remainder of the folders are dedicated to reproducing the experiments of our article:

+ `utils` contains additional scripts for dataset creation, result gathering.
+ `analysis` contains the necessary script to re-obtain the contents of the figures.
+ `configs` contains a set of configuration files for each different experiment.

## Other scripts details

There are a couple scripts in the `utils` folder that are useful for the analysis or the snakemake pipeline during experiments.

+ `compute_distances`
+ `create_dataset`
+ `extract_common_features`
+ `extract_logreg_feature_history`
+ `extract_mnist_feature_importance`
+ `fetch_mnist`
+ `fetch_openml`
+ `merge_clustering`
+ `merge_clusterings_v2`
+ `merge_selections`
+ `retrieve_optimal_solutions`

## Redoing experiments

If you are interested in replicating some experiments, please refer to the main file `How to redo all experiments.md` which lists the step-by-step command lines for all experiments.
# Synthetic datasets

## Getting results

Scripts required: `create_dataset`, `compute_distances`

### Synthetic datasets

```
snakemake all --cores 4 --configfile configs/synthetic/static/celeux_1s1.yml
```

```
snakemake all --cores 4 --configfile configs/synthetic/static/celeux_1s2.yml
```

```
snakemake all --cores 4 --configfile configs/synthetic/static/celeux_1s3.yml
```

```
snakemake all --cores 4 --configfile configs/synthetic/static/celeux_1s4.yml
```

```
snakemake all --cores 4 --configfile configs/synthetic/static/celeux_1s5.yml
```

### Openml

Scripts required: `fetch_openml`

```
snakemake all --cores 4 --configfile configs/openml/heart_statlog_static.yml
```
```
snakemake all --cores 4 --configfile configs/openml/us_congress.yml
```

### MNIST

#### Standard

We start by preparing the dataset
```commandline
mkdir -p datasets/mnist/ distances/mnist/
python utils/fetch_mnist.py --data_path path/to/your/data --output_file datasets/mnist/mnist.csv
python utils/compute_distances.py --dataset datasets/mnist/mnist.csv --kernel --output distances/mnist/mnist_kernel.csv
```

Then, we can run the Sparse GEMINI algorithm on this subset of MNIST.

```commandline
python sparse_gemini/main.py --gemini mmd --csv datasets/mnist/mnist.csv --mode ova --output_path results/mnist/mmd  --static_metric euclidean --use_cuda --dims 1200 1200 --feature_threshold 50  --dropout 0.05 --patience 10 --epochs 100 --batch_size 1000 -K 10 --lambda_start 40 --lambda_multiplier 1.05 --seed 0
```


#### Variations

We start by preparing the dataset

```
mkdir -p datasets/mnist_br
https://web.archive.org/web/20180519112150/http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip
mv mnist_background_random.zip datasets/mnist_br
unzip datasets/mnist_br/mnist_random_background.zip -d datasets/mnist_br/
python utils/preprocess_mnist_br.py --input_file datasets/mnist_br/mnist_background_random_train.amat --output_file datasets/mnist_br/train.csv --targets datasets/mnist_br/targets.csv
```

For running:

```commandline
python sparse_gemini/main.py --gemini mmd --csv datasets/mnist_background_random/train_data.csv --mode ova --output_path results/mnist_background_random/mmd  --static_metric euclidean --use_cuda --dims 1200 1200 --feature_threshold 50  --dropout 0.05 --patience 10 --epochs 100 --batch_size 1000 -K 10 --lambda_start 40 --lambda_multiplier 1.05 --seed 0
```
### Prostate BCR

This is only the MMD version. You just need to change if using Wasserstein and ovo mode.
Do not forget to change as well to compute the distance.

We start by preparing the dataset and computing the distances / kernel.
```commandline
wget https://raw.githubusercontent.com/ArnaudDroitLab/prostate_BCR_prediction/main/training_data_three_Pca_cohorts.tsv
python utils/preprocess_bcr.py --input_file training_data_three_Pca_cohorts.tsv --output_file datasets/real_datasets/prostate_bcr/preprocessed_data.csv --targets datasets/real_datasets/prostate_bcr/targets.csv --set_targets datasets/real_datasets/prostate_bcr/set_targets.csv
mkdir distances/real_datasets/prostate_bcr -p
python utils/compute_distances.py --dataset datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output distances/real_datasets/prostate_bcr/prostate_bcr_kernel.csv --kernel 
python utils/compute_distances.py --dataset datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output distances/real_datasets/prostate_bcr/prostate_bcr_distances.csv
mkdir -p results/prostate_bcr/{mlp,linear}/{mmd,wasserstein}_{2,3}
```

Then we can run all experiments

MLP MMD, 2 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/mlp/mmd_2/run_${i} --gemini mmd --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_kernel.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 2 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02; done
```

MLP MMD 3 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/mlp/mmd_3/run_${i} --gemini mmd --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_kernel.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 3 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02; done
```


MLP Wasserstein 2 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/mlp/wasserstein_2/run_${i} --gemini wasserstein --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_distances.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 2 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02; done
```

MLP Wasserstein 3 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/mlp/wasserstein_3/run_${i} --gemini wasserstein --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_distances.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 3 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02; done
```

Linear MMD, 2 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/linear/mmd_2/run_${i} --gemini mmd --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_kernel.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 2 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02 -M 0; done
```

Linear MMD 3 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/linear/mmd_3/run_${i} --gemini mmd --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_kernel.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 3 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02 -M 0; done
```

Linear Wasserstein 2 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/linear/wasserstein_2/run_${i} --gemini wasserstein --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_distances.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 2 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02 -M 0 ; done
```

Linear Wasserstein 3 clusters
```commandline
for i in $(seq 1 5); do python -u sparse_gemini/main.py --csv datasets/real_datasets/prostate_bcr/preprocessed_data.csv --output_path results/prostate_bcr/linear/wasserstein_3/run_${i} --gemini wasserstein --mode ova --metric_file distances/real_datasets/prostate_bcr/prostate_bcr_distances.csv --use_cuda --epochs 50 --batch_size 57 --feature_threshold 400 --dims 100 --num_clusters 3 --dropout 0 --lambda_start 1 --lambda_multiplier 1.02 -M 0; done
```

## Running the baselines

```commandline
cd baseline
snakemake --configfile configs/openml.yml --cores 16 -p
snakemake --configfile configs/celeux_s1.yml --cores 16 -p
snakemake --configfile configs/celeux_s2.yml --cores 16 -p
```

For analysis: `analysis/Baseline.Rmd`

## Analysing results and getting figures

### Generating summaries

All the figure generations script will need some ready-made csvs containing the summary of all runs. To that end, we will use the following scripts:

+ `extract_weight_history`: for extracting the history of the weights of the skip connection for a particular run
+ `merge_clusterings`: to compute the clustering scores against (supervised) targets
+ `merge_selections`: to compute the variable selections scores against (supervised) targets
+ `retrieve_optimal_solutions`: to extract for every run the clustering corresponding to 90\% of the top GEMINI while using the fewest variables

We start by extracting the performances for all synthetic runs:

```commandline
mkdir analysis/summaries
# Take examples of weights history
python utils/extract_weight_history.py --input_folder results/synthetic/celeux_1s5_static/mlp/wasserstein/ova/run_0/ --output_file analysis/summaries/example_weights_1s5.csv
python utils/extract_weight_history.py --input_folder results/synthetic/celeux_s2_static/mlp/wasserstein/ova/run_0/ --output_file analysis/summaries/example_weights_s2.csv
# Combine the clustering scores
for folder in $(ls results/synthetic); do python utils/merge_clusterings.py --input_folder results/synthetic/${folder} --output_file analysis/summaries/${folder}_clustering.csv --data_folder datasets/synthetic; done
# Compute variable expectation scores
for scenario in $(seq 1 5); do for type in dynamic static; do python utils/merge_selections.py --input_folder results/synthetic/celeux_1s${scenario}_${type} --lambda_start 1 --lambda_multiplier 1.05 --expected_F 5 --output_file analysis/summaries/celeux_1s${scenario}_${type}_features.csv; python utils/retrieve_optimal_solutions.py --input_folder results/synthetic/celeux_1s${scenario}_${type} --output_file analysis/summaries/celeux_1s${scenario}_${type}_optimal.csv --data_folder datasets/synthetic/ --selection_threshold 0.9 --expected_F 5 --lambda_multiplier 1.05 --lambda_start 1 ; done; done
for type in dynamic static; do python utils/merge_selections.py --input_folder results/synthetic/celeux_s2_${type} --lambda_start 1 --lambda_multiplier 1.05 --expected_F 2 --output_file analysis/summaries/celeux_s2_${type}_features.csv; python utils/retrieve_optimal_solutions.py --input_folder results/synthetic/celeux_s2_${type} --output_file analysis/summaries/celeux_s2_${type}_optimal.csv --data_folder datasets/synthetic/ --selection_threshold 0.9 --expected_F 2 --lambda_multiplier 1.05 --lambda_start 1; done
```

We do the same thing for the openml datasets except there is no expectation regarding the number of selected features.

```commandline
for folder in $(ls results/openml); do python utils/merge_clusterings.py --input_folder results/openml/${folder} --output_file analysis/summaries/${folder}_clustering.csv --data_folder datasets/openml; python utils/merge_selections.py --input_folder results/openml/${folder} --lambda_start 1 --lambda_multiplier 1.1 --expected_F 5 --output_file analysis/summaries/${folder}_features.csv; python utils/retrieve_optimal_solutions.py --input_folder results/openml/${folder} --output_file analysis/summaries/${folder}_optimal.csv --data_folder datasets/openml/ --selection_threshold 0.9 --expected_F 5 --lambda_multiplier 1.1 --lambda_start 1 --export_feature_choice; done
```

### MNIST

```commandline
python utils/evaluate_clusterings.py --input_folder results/mnist --targets datasets/mnist/mnist_targets.csv --output_file analysis/summaries/mnist_plain_clustering.csv
python utils/evaluate_clusterings.py --input_folder results/mnist_background_random/ --targets datasets/mnist_br/targets.csv --output_file analysis/summaries/mnist_br_clustering.csv
```

### Prostate

```commandline
# Evaluate using the targets regarding BCR
for architecture in mlp linear; do echo $architecture; for folder in mmd_2 mmd_3 wasserstein_2 wasserstein_3; do echo $folder; for i in $(seq 1 5); do echo $i; python utils/evaluate_clusterings.py --input_folder results/prostate_bcr/${architecture}/${folder}/run_${i} --targets datasets/real_datasets/prostate_bcr/targets.csv --output_file analysis/summaries/bcr_${architecture}_${folder}_${i}_clustering.csv; done; done; done
# Evaluating using the data source as targets
for architecture in mlp linear; do echo $architecture; for folder in mmd_2 mmd_3 wasserstein_2 wasserstein_3; do echo $folder; for i in $(seq 1 5); do echo $i; python utils/evaluate_clusterings.py --input_folder results/prostate_bcr/${architecture}/${folder}/run_${i} --targets datasets/real_datasets/prostate_bcr/set_targets.csv --output_file analysis/summaries/bcr_${architecture}_${folder}_${i}_set_clustering.csv; done; done; done
```


### Generating figures

### Synthetic results

```commandline
mkdir -p figs/synthetic
```

Then, you can run the R markdown file `analysis/Synthetic.Rmd` to generate all figures.

### Openml

```commandline
mkdir -p figs/openml
```

The tables were generated through the R markdown file `analysis/Openml.Rmd`

### MNIST

The tables were generated through the R markdown file `analysis/Mnist.Rmd`

```commandline
python utils/extract_mnist_feature_importance.py --input_file results/mnist/mmd/feature_importances.csv --lambda_start 40 --lambda_multiplier 1.1 --output_file analysis/figs/mnist/mnist_plain_features.png
python utils/extract_mnist_feature_importance.py --input_file results/mnist_background_random/mmd/feature_importances.csv --lambda_start 40 --lambda_multiplier 1.1 --output_file analysis/figs/mnist/mnist_br_features.png
```
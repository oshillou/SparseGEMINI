from arguments import get_args
from data_utils import *
from train_utils import perform_path, compute_feature_importances, build_selection_history, build_clustering_results
from io_utils import export_csv, create_output
from unsupervised_lassonet import Model
import torch


def main():
    args = get_args()
    create_output(args)

    if args.seed != -1:
        print(f"Fixing the seed to {args.seed}")
        torch.random.manual_seed(args.seed)

    # First, load the dataset
    print("Loading the dataset")
    X = get_dataset(args)
    if args.dynamic_metric is not None:
        print(f"Running in dynamic mode with {args.dynamic_metric} metric")
        affinity_fct = get_affinity_function(args)
        dataset = DynamicAffinityDataset(X, affinity_fct)
    elif args.static_metric is not None:
        print(f"Running on static mode {args.static_metric} metric")
        affinity_fct = get_affinity_function(args)
        dataset = DynamicAffinityDataset(X, affinity_fct)
    else:
        print(f"Running on fixed mode with metric from {args.metric}")
        D = get_affinity(args)
        dataset = AffinityDataset(X, D)

    print("Creating the model")
    model = Model(dataset.get_input_shape(), *args.dims, args.num_clusters, dropout=args.dropout, M=args.M)

    # Now, we can walk the path of regularisation
    print("Starting all trainings")
    path_history = perform_path(args, dataset, model)

    print("Computing feature importance")
    feature_importance = compute_feature_importances(path_history)
    print(feature_importance)

    export_csv(args, feature_importance, "feature_importances.csv")

    feature_history = build_selection_history(path_history)
    export_csv(args, feature_history, "feature_history.csv")

    print("Computing the final clustering file")
    clustering_results = build_clustering_results(args, model, dataset, path_history)
    export_csv(args, clustering_results, "clustering.csv")


if __name__ == '__main__':
    main()

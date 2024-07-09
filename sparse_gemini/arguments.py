import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=-1,
                        help="The value of the random seed. If set to -1, no seed will be used")
    io_group = parser.add_argument_group("Input Output", "Parameters related to input of data and output")
    data_group = io_group.add_mutually_exclusive_group()
    data_group.add_argument("--csv", type=str,
                            help="The path to the file containing your matrix data in a csv format. The file must not "
                                 "contain indices and column headers are not required.")
    data_group.add_argument("--data", type=str, help="The name of a standard Deep learning dataset",
                            choices=["mnist", "fashionmnist"])
    io_group.add_argument("--data_path", type=str, default="./data",
                          help="If you use one deep learning dataset through the --data option, use this option"
                               "in conjunction to specify where to look up and save the files.")
    io_group.add_argument("--output_path", type=str, default='./',
                          help="The path where the output files will be produced.")

    gemini_group = parser.add_argument_group("GEMINI", "Parameters related to the GEMINI to optimise")
    gemini_group.add_argument("--gemini", type=str, default="mmd", choices=["mmd", "wasserstein"],
                              help="The choice of the statistical distance at the core of the GEMINI. For ovo/ova "
                                   "selection, use the --ovo option.")
    gemini_group.add_argument("--mode", default="ova", choices=["ova", "ovo"],
                              help="Specify this option if you want to work with a one-vs-one gemini. If unspecified, "
                                   "the algorithm will run using the one-vs-all setup.")
    metric_group = gemini_group.add_mutually_exclusive_group()
    metric_group.add_argument("--metric_file", type=str,
                              help="The metric to use in combination with the gemini. It is up to you to verify that "
                                   "the provided metric respects the rules of a distance / kernel before using it "
                                   "respectively with Wasserstein or MMD.",
                              dest="metric")
    metric_group.add_argument("--static_metric",type=str,
                              help="The metric that will be computed on the data space consisting of all "
                                   "features.", choices=["sqeuclidean", "euclidean"])
    metric_group.add_argument("--dynamic_metric", type=str,
                              help="The metric that will be computed on the data space consisting of the remaining "
                                   "selected features.", choices=["sqeuclidean", "euclidean"])

    training_group = parser.add_argument_group("Training", "Parameters related to the model training")
    training_group.add_argument("--use_cuda", default=False, action='store_true',
                                help="Specify this option to benefit from GPU acceleration if available.")
    training_group.add_argument("--epochs", type=int, default=100,
                                help="Number of iterations to let the model converge")
    training_group.add_argument("--batch_size", "-B", type=int, dest="batch_size", default=-1,
                                help="The batch size for training. It needs to be comprised between 1 and the shape "
                                     "of the number of samples in the dataset. If you set it to -1 (default), "
                                     "the training will be performed on the entire data at once at every epoch")
    for prefix in ["init", "path"]:
        training_group.add_argument(f"--{prefix}_optim", default="adam" if prefix == "init" else "sgd",
                                    choices=["adam", "sgd"], help=f"The optimiser for the {prefix} step")
        training_group.add_argument(f"--{prefix}_lr", default=1e-3,
                                    help=f"Model learning rate for optimiser during the {prefix} step")

    training_group.add_argument("--tol", type=float, default=0.99,
                                help="The tolerance ratio between two consecutive epochs of loss for early stopping")
    training_group.add_argument("--patience", type=int, default=10,
                                help="The number of allowed epochs without any progression of the objective to "
                                     "trigger the early stopping")
    training_group.add_argument("--feature_threshold", type=int, default=0,
                                help="The minimal number of features to reach before stopping")

    model_group = parser.add_argument_group("Model", "Parameters related to the model")
    model_group.add_argument("--dims", type=int, default=[100], nargs="+",
                             help="The dimension of each hidden layer before the final clustering layer")
    model_group.add_argument("--num_clusters", "-K", dest="num_clusters", default=10, type=int,
                             help="The number of clusters you want to find *at best*.")
    model_group.add_argument("--dropout", type=float, default=None, help="The ratio for dropout in the LassoNet model")

    reg_group = parser.add_argument_group("Regularisations and path",
                                          "Optional parameters to complete the training with some regularisations")
    reg_group.add_argument("--lambda_start", type=float, default=1e-4,
                           help="The starting value of the lambda parameter")
    reg_group.add_argument("--lambda_multiplier", type=float, default=1.02,
                           help="The factor that will multiply the penalty term at each step of the path")
    reg_group.add_argument("-M", type=float, default=10,
                           help="I can't remember the explanation, but it is in the paper")

    return parser.parse_args()

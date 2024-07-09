import argparse
from os.path import exists as pexists, join as pjoin
import pandas as pd
from glob import glob
import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./output.csv")

    args = parser.parse_args()
    assert pexists(args.input_folder), f"Please provide a valid input folder. {args.input_folder} does not exist."

    return args


def retrieve_run_information(args, folder):
    # We are in a folder where there will be a couple of models saved
    # We first get the clustering.csv file which describes all of these models
    csv = pd.read_csv(pjoin(folder, "clustering.csv"))
    csv = csv.loc[:, ["F", "lambda"]]

    norms = []
    for f in csv.F:
        model_weights = torch.load(pjoin(folder, f"best_model_{f}_features"))
        skip_weights = model_weights["skip.weight"]
        skip_norm = torch.linalg.norm(skip_weights, dim=0)
        norms += [{f"F_{i}": skip_norm[i].item() for i in range(len(skip_norm))}]
    norms = pd.DataFrame(norms)

    # No need to perform join since the features were looped in the same order as the rows
    # of the original csv
    return pd.concat([csv, norms],axis=1)


def explore_all_runs(args, folder):
    if pexists(pjoin(folder, "clustering.csv")):
        # We found a final folder which contains a model
        # We can retrieve informations
        return [retrieve_run_information(args, folder)]
    # There are still subfolders to explore
    subfolders = glob(pjoin(folder, "*"))
    total_results = []
    for subfolder in subfolders:
        total_results.extend(explore_all_runs(args, subfolder))
    return total_results


def fuse_results(args):
    total_results = explore_all_runs(args, args.input_folder)

    return pd.concat(total_results, axis=0, ignore_index=True)


def main():
    args = get_args()

    feature_weights_csv = fuse_results(args)

    feature_weights_csv.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()

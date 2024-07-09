import argparse
import os
from os.path import join as pjoin, exists as pexists
from glob import glob
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import pandas as pd
from scipy.optimize import linear_sum_assignment as lsa
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./")
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--many2one", action="store_true", default=False)

    args = parser.parse_args()
    assert pexists(args.input_folder), f"Please provide a valid input folder. {args.input_folder} does not exist."

    return args


def unsupervised_accuracy(y_true, y_pred):
    cmatrix = confusion_matrix(y_true, y_pred)

    r, c = lsa(cmatrix, maximize=True)

    return cmatrix[r, c].sum() / cmatrix.sum()


def many2one_score(y_true, y_pred):
    cmatrix = confusion_matrix(y_true, y_pred)
    indicator = cmatrix.argmax(0)

    y_pred = indicator[y_pred]

    ari = adjusted_rand_score(y_true)
    acc = unsupervised_accuracy(y_true, y_pred)
    return ari, acc


def one2one_score(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)

    acc = unsupervised_accuracy(y_true, y_pred)

    return ari, acc


def retrieve_run_information(args, folder):
    clusterings = pd.read_csv(pjoin(folder, "clustering.csv"))

    # Find the matching target file
    print(f"Analysing {folder}")
    data_name = args.input_folder.split(os.sep)[-1]
    run_id = folder.split("_")[-1]
    targets = pjoin(args.data_folder, data_name + "_" + run_id + "_targets.csv")
    targets = pd.read_csv(targets).to_numpy().reshape((-1))

    clustering_columns = [x for x in clusterings.columns if x[:2] == "c_"]

    if targets is not None:
        aris = []
        accuracies = []
        for _, row in clusterings.iterrows():
            y_pred = row[clustering_columns].to_numpy().reshape((-1))

            if args.many2one:
                scores = many2one_score(targets, y_pred)
            else:
                scores = one2one_score(targets, y_pred)

            aris += [scores[0]]
            accuracies += [scores[1]]
        clusterings["ARI"] = aris
        clusterings["ACC"] = accuracies

    # Drop the clustering column
    clusterings.drop(clustering_columns, axis=1, inplace=True)

    folder_names = folder.split(os.sep)
    mode = folder_names[-2]
    gemini = folder_names[-3]
    architecture = folder_names[-4]

    clusterings["mode"] = mode
    clusterings["distance"] = gemini
    clusterings["model"]=architecture

    return clusterings


def explore_all_runs(args, folder):
    if pexists(pjoin(folder, "clustering.csv")):
        return [retrieve_run_information(args, folder)]
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
    total_results = explore_all_runs(args, args.input_folder)
    complete_csv = pd.concat(total_results, axis=0, ignore_index=True)

    complete_csv.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()

import argparse
from os.path import exists as pexists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--input_file",type=str,required=True)
    parser.add_argument("--lambda_start", type=float, default=1.0)
    parser.add_argument("--lambda_multiplier",type=float,default=1.1)
    parser.add_argument("--output_file",type=str,default="./output.svg")

    args=parser.parse_args()
    assert pexists(args.input_file), f"Please provide a valid input file. {args.input_file} does not exist."

    return args

def retrieve_run_information(args):
    feature_importance = pd.read_csv(args.input_file).to_numpy()

    # Replace the feature importance inf with lambda_mult * second max importance
    best_max=feature_importance[~np.isinf(feature_importance)].max()
    feature_importance[np.isinf(feature_importance)]=args.lambda_multiplier*best_max

    # Rescale according to the logarithmic scale
    X_scaled = (np.log(feature_importance)-np.log(args.lambda_start))/np.log(args.lambda_multiplier)

    # Now, scale again between 0 and 1
    X_scaled = (X_scaled-X_scaled.min())/(X_scaled.max()-X_scaled.min())

    # We thus have a linear scale [0,1] where 0 = lambda = lambda_min and 1=lambda_start*lambda_mult**n_max
    return X_scaled


def main():
    args=get_args()
    print(f"Fetching results from {args.input_file} and scaling them")
    X = retrieve_run_information(args)

    print(f"Exporting to {args.output_file}")
    fig=plt.figure()
    plt.imshow(X.reshape((28,28)), cmap='seismic',vmin=0,vmax=1)
    plt.axis('off')
    plt.savefig(args.output_file, bbox_inches="tight")

    print("Finished")


if __name__=='__main__':
    main()

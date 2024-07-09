import pandas as pd
import argparse
from os.path import exists as pexists


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True,
                        help="The dataset file containing the mnist variation in .amat format")
    parser.add_argument("--output_file", type=str, default="data.csv")
    parser.add_argument("--targets", type=str, default="targets.csv")

    args = parser.parse_args()

    assert pexists(args.input_file), f"The provided input file does not exist: {args.input_file}"

    return args


def main():
    args = get_args()

    print("Loading dataset")
    dataset = pd.read_csv(args.input_file, sep="   ", header=None)
    print(f"Shape is {dataset.shape}\nDropping useless columns")

    data = dataset[dataset.columns[-1]]
    targets = dataset[dataset.columns[-1]].astype(int)

    print(f"Exporting preprocessed dataset to {args.output_file}")
    data.to_csv(args.output_file, index=False)

    print(f"Exporting targets to {args.targets}")
    targets.to_csv(args.targets, index=False)

    print("Finished")


if __name__ == "__main__":
    main()


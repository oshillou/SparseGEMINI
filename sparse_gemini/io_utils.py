import os
from os.path import join, exists
from glob import glob
import pandas as pd
import torch


def create_output(args):
    if exists(args.output_path):
        assert len(os.listdir(
            args.output_path)) == 0, f"Please provide an empty folder for the output files or a non-existing folder"
    os.makedirs(args.output_path, exist_ok=True)


def load_csv(csv_name):
    df = pd.read_csv(csv_name, sep=',', index_col=None)

    return df.to_numpy()


def save_state_dict(args, state_dict, name):
    path_to_file = join(args.output_path, name)
    torch.save(state_dict, path_to_file)


def load_model(args, filename, model):
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    return model


def export_csv(args, data, name):
    path_to_file = join(args.output_path, name)
    pd.DataFrame(data).to_csv(path_to_file, index=False)


def get_all_models(args):
    all_models = glob(join(args.output_path, "best_model_*_features"))
    return all_models

import argparse
import os
from os.path import join as pjoin, exists as pexists
from glob import glob
import pandas as pd
import numpy as np

def get_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--input_folder",type=str,required=True)
    parser.add_argument("--lambda_start",type=float,default=1e-2)
    parser.add_argument("--lambda_multiplier",type=float,default=1.1)
    parser.add_argument("--expected_F",type=int,default=2)
    parser.add_argument("--output_file",type=str,default="./")

    args=parser.parse_args()
    assert pexists(args.input_folder), f"Please provide a valid input folder. {args.input_folder} does not exist."

    return args

def retrieve_run_information(args,folder):
    history=pd.read_csv(pjoin(folder, "feature_history.csv"))

    expectations = np.arange(history.shape[1]) < args.expected_F
    vser=(history!=expectations).mean(1)
    cvr=(history[[str(x) for x in range(args.expected_F)]]).mean(1)
    history["F"]=history.sum(1)
    history["VSER"]=vser
    history["CVR"]=cvr
    history=history[["F","VSER","CVR"]]

    num_runs=history.shape[0]-1
    lambdas=np.array([0]+list(args.lambda_start+args.lambda_multiplier**np.linspace(0,num_runs-1,num=num_runs)))
    history["Lambda"]=lambdas

    folder_names = folder.split(os.sep)
    mode=folder_names[-2]
    gemini=folder_names[-3]
    architecture = folder_names[-4]

    history["mode"]=mode
    history["distance"]=gemini
    history["model"]=architecture
    return history

def explore_all_runs(args, folder):
    if pexists(pjoin(folder,"feature_history.csv")):
        return [retrieve_run_information(args,folder)]
    subfolders=glob(pjoin(folder,"*"))
    total_results=[]
    for subfolder in subfolders:
        total_results.extend(explore_all_runs(args,subfolder))
    return total_results

def main():
    args=get_args()
    print(f"Fetching results from {args.input_folder}")
    total_results = explore_all_runs(args, args.input_folder)
    complete_csv=pd.concat(total_results,axis=0,ignore_index=True)
    print(f"Fetched a total of {complete_csv.shape[0]} entries and {complete_csv.shape[1]} variables")

    print(f"Exporting to {args.output_file}")
    complete_csv.to_csv(args.output_file, index=False)

if __name__=='__main__':
    main()
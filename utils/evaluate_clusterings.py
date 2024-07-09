import argparse
import os
from os.path import join as pjoin, exists as pexists
from glob import glob
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.optimize import linear_sum_assignment as lsa

def get_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--input_folder",type=str,required=True)
    parser.add_argument("--output_file",type=str,default="./")
    parser.add_argument("--targets",type=str,default="")
    parser.add_argument("--many2one",action="store_true",default=False)

    args=parser.parse_args()
    assert pexists(args.input_folder), f"Please provide a valid input folder. {args.input_folder} does not exist."

    return args

def unsupervised_accuracy(y_true,y_pred):
    cmatrix=confusion_matrix(y_true,y_pred)

    r,c=lsa(cmatrix,maximize=True)

    return cmatrix[r,c].sum()/cmatrix.sum()

def many2one_score(y_true,y_pred):
    y_pred=y_pred.astype(int)
    # To overcome empty clusters, we encode again the prediction
    y_pred=LabelEncoder().fit_transform(y_pred)
    cmatrix=confusion_matrix(y_true,y_pred)
    indicator=cmatrix.argmax(0)

    y_pred=indicator[y_pred.astype(int)]

    ari=adjusted_rand_score(y_true, y_pred)
    acc = unsupervised_accuracy(y_true,y_pred)
    return ari, acc

def one2one_score(y_true,y_pred):

    ari = adjusted_rand_score(y_true,y_pred)

    acc = unsupervised_accuracy(y_true, y_pred)

    return ari, acc

def retrieve_run_information(args, folder,targets=None):
    clusterings=pd.read_csv(pjoin(folder, "clustering.csv"))

    clustering_columns=[x for x in clusterings.columns if x[:2]=="c_"]

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
    clusterings.drop(clustering_columns,axis=1,inplace=True)

    mode=folder.split(os.sep)[-2]
    gemini=folder.split(os.sep)[-3]

    clusterings["mode"]=mode
    clusterings["distance"]=gemini

    return clusterings

def explore_all_runs(args, folder,targets=None):
    if pexists(pjoin(folder,"clustering.csv")):
        return [retrieve_run_information(args,folder,targets)]
    subfolders=glob(pjoin(folder,"*"))
    total_results=[]
    for subfolder in subfolders:
        total_results.extend(explore_all_runs(args,subfolder,targets))
    return total_results

def fuse_results(args,targets=None):
    total_results=explore_all_runs(args,args.input_folder,targets)

    return pd.concat(total_results,axis=0,ignore_index=True)

def main():
    args=get_args()

    if args.targets!="":
        targets=pd.read_csv(args.targets).to_numpy().reshape((-1))
    else:
        targets=None
    total_results = explore_all_runs(args,args.input_folder, targets)
    complete_csv=pd.concat(total_results,axis=0,ignore_index=True)

    complete_csv.to_csv(args.output_file, index=False)

if __name__=='__main__':
    main()
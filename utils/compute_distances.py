import argparse
import pandas as pd
from os.path import exists as pexists
from sklearn import preprocessing, metrics
import numpy as np

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset",type=str,required=True)
	parser.add_argument("--output",type=str,default="./out.csv")
	parser.add_argument("--kernel",action='store_true')
	parser.add_argument("--variables",type=str,default="cont", choices=["cont","cat"])

	args=parser.parse_args()

	assert pexists(args.dataset), f"{args.dataset} does not exist"

	return args

def compute_continuous_distance(X, kernel=False):
	if kernel:
		return X@X.T
	else:
		return metrics.pairwise_distances(X,metric='euclidean')

def compute_categorical_distance(X,kernel=False):
	Y=[]
	for x in X.T:
		Y+=[preprocessing.OneHotEncoder(sparse=False).fit_transform(x.reshape((-1,1)))]
	one_hot_X=np.concatenate(Y,axis=1)

	return compute_continuous_distance(one_hot_X,kernel=kernel)

def main():
	# Retrieve arguments
	args=get_args()

	# Get the dataframe
	df=pd.read_csv(args.dataset)

	if args.variables=="cont":
		X=compute_continuous_distance(df.to_numpy(),kernel=args.kernel)
	else:
		X=compute_categorical_distance(df.to_numpy(),kernel=args.kernel)

	pd.DataFrame(X).to_csv(args.output,sep=',',index=False)


if __name__=='__main__':
	main()

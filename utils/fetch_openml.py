from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import argparse
import pandas as pd
from os.path import join as pjoin

def get_args():
	parser=argparse.ArgumentParser()
	parser.add_argument("--output_path",type=str,default="./out.csv")
	parser.add_argument("--export_targets",action="store_true",default=False)

	parser.add_argument("--name",type=str,required=True, choices=["miceprotein","dataset_53_heart-statlog", "vote"])

	subparsers=parser.add_subparsers(dest="dataset")

	return parser.parse_args()

def preapre_mice(X,y):
	X=SimpleImputer().fit_transform(X)
	X=StandardScaler().fit_transform(X)
	y=LabelEncoder().fit_transform(y)

	return X,y

def prepare_congress(X,y):
	y=LabelEncoder().fit_transform(y)
	X=X.to_numpy()
	X[pd.isna(X)]=0
	X[X=='y']=1
	X[X=='n']=-1
	
	return X,y

def prepare_heart(X,y):
	y = LabelEncoder().fit_transform(X["class"])
	X = X.loc[:,X.columns!="class"]
	X = StandardScaler().fit_transform(X)
	return X,y

def main():
	args=get_args()

	X,y=fetch_openml(name=args.name,return_X_y=True)
	prepare_fct = id
	if args.name=="miceprotein":
		prepare_fct=prepare_mice
	elif args.name=="vote":
		prepare_fct=prepare_congress
	elif args.name=="dataset_53_heart-statlog":
		prepare_fct=prepare_heart
	X,y=prepare_fct(X,y)
	pd.DataFrame(X).to_csv(args.output_path, index=False)

	if args.export_targets:
		target_file=args.output_path.replace(".csv","_targets.csv")
		pd.DataFrame(y).to_csv(target_file,index=False)

if __name__=='__main__':
	main()

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import argparse
from os.path import exists as pexists

def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--input_file",type=str,required=True,help="The dataset file for the prostate cancer in tsv format")
	parser.add_argument("--output_file",type=str,default="data.csv")
	parser.add_argument("--targets",type=str,default="targets.csv")
	parser.add_argument("--set_targets", type=str, default="set_targets.csv")
	
	args = parser.parse_args()
	
	assert pexists(args.input_file), f"The provided input file does not exist: {args.input_file}"
	
	return args

def main():
	args = get_args()
	
	print("Loading dataset")
	dataset = pd.read_csv(args.input_file, sep="\t")
	print(f"Shape is {dataset.shape}\nDropping useless columns")
	
	targets = dataset["BCR_60"]
	set_targets = dataset["Set"].to_numpy()
	set_targets = OneHotEncoder(sparse=False).fit_transform(set_targets.reshape((-1,1))).argmax(1)
	dataset = dataset.drop(["patient","BCR_60","gleason","grade","study","Set"],axis=1)
	
	print(f"New shape is {dataset.shape}\nScaling")
	dataset_scaled=pd.DataFrame(StandardScaler().fit_transform(dataset))
	dataset_scaled.columns=dataset.columns
	
	print(f"Exporting preprocessed dataset to {args.output_file}")
	dataset_scaled.to_csv(args.output_file,index=False)
	
	print(f"Exporting targets to {args.targets}")
	targets.to_csv(args.targets,index=False)

	print(f"Exporting set targets to {args.targets}")
	pd.DataFrame(set_targets, columns=["Set"]).to_csv(args.set_targets, index=False)
	
	print("Finished")

if __name__=="__main__":
	main()
	

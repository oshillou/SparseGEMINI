from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

import pandas as pd
import argparse


def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--output_file",type=str,default="./mnist.csv")
	parser.add_argument("--data_path",type=str,default="./data/")

	return parser.parse_args()

def main():
	args = get_args()

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.5],[0.5]),
		transforms.Lambda(lambda x: torch.flatten(x))])

	print("Fetching mnist")
	dataset = datasets.MNIST(root=args.data_path, train=True, transform=transform)

	print("Taking samples")
	sub_samples = next(iter(DataLoader(dataset, batch_size=12000, shuffle=False)))

	print(f"Exporting to {args.output_file}")
	pd.DataFrame(sub_samples[0]).to_csv(args.output_file,index=False)

	target_file=args.output_file.replace(".csv","_targets.csv")
	print(f"Exporting targets to {target_file}")
	pd.DataFrame(sub_samples[1]).to_csv(target_file,index=False)


if __name__=="__main__":
	main()
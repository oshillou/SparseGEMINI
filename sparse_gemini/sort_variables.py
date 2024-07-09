import argparse
import os
from os.path import join as pjoin
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="./output", type=str)
args = parser.parse_args()

all_variables_files = list(filter(lambda x: "variables" in x, os.listdir(args.path)))
all_variables = [np.load(pjoin(args.path, x)) for x in all_variables_files]
lambdas = [float(x[:-4].split("_")[1]) for x in all_variables_files]
lambda_order = sorted(range(len(lambdas)), key=lambda x: lambdas[x])

top_n_variables = 0
for i in lambda_order:
    print(f"Lambda: {lambdas[i]}:\t{len(all_variables[i])} variables")
    top_n_variables = max(top_n_variables, len(all_variables[i]))

# Now, generate the figure of the used variables
total_used = np.zeros(top_n_variables)
for variable in np.concatenate(all_variables):
    total_used[variable] += 1

print(total_used)

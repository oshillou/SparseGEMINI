wildcard_constraints:
	gemini="(wasserstein|mmd)",
	mode="(ova|ovo)",
	i="\d+",
	dist_type="(kernel|distance)"

def get_source(wildcards):
	if config['data_source']=="None":
		return "Snakefile"
	else:
		return config['data_source']

def get_sparse_gemini_inputs(wildcards):
	if config['dynamic']:
		return [config["data_path"]+"/{data_name}_{i}.csv"]
	else:
		distance=f"{config['distance_path']}/{wildcards.data_name}_{wildcards.i}_"
		if wildcards.gemini == 'wasserstein':
			distance=distance + 'distance.csv'
		else:
			distance=distance + 'kernel.csv'
		return [config["data_path"]+"/{data_name}_{i}.csv",distance]

def get_sparse_gemini_distance_param(wildcards):
	if config['dynamic']:
		return " --dynamic_metric euclidean"
	else:
		return " --metric_file "+get_sparse_gemini_inputs(wildcards)[1]

def get_kernel_option(wildcards):
	if wildcards.dist_type=="kernel":
		return "--kernel"
	return ""

rule all:
	input:
		expand("results/{data_name}/{model}/{gemini}/{mode}/run_{i}/clustering.csv", data_name=config['data_name'], model=["mlp","linear"], gemini=config['gemini'], mode=config['mode'], i=range(config['num_runs']))

rule run_mlp_sparse_gemini:
	input:
		get_sparse_gemini_inputs
	output:
		"results/{data_name}/mlp/{gemini}/{mode}/run_{i}/clustering.csv"
	params:
	    output_dir=directory("results/{data_name}/mlp/{gemini}/{mode}/run_{i}"),
	    affinity=get_sparse_gemini_distance_param,
	log:
		out="logs/{data_name}/mlp/{gemini}_{mode}_run_{i}.out",
		err="logs/{data_name}/mlp/{gemini}_{mode}_run_{i}.err"
	shell:
		"python -u sparse_gemini/main.py --gemini {wildcards.gemini} --csv {input[0]} --mode {wildcards.mode} " + \
		"--output_path {params.output_dir} {params.affinity} " + config["lasso_vars"] + " > {log.out} 2> {log.err}"

rule run_linear_sparse_gemini:
	input:
		get_sparse_gemini_inputs
	output:
		"results/{data_name}/linear/{gemini}/{mode}/run_{i}/clustering.csv"
	params:
	    output_dir=directory("results/{data_name}/linear/{gemini}/{mode}/run_{i}"),
	    affinity=get_sparse_gemini_distance_param,
	log:
		out="logs/{data_name}/linear/{gemini}_{mode}_run_{i}.out",
		err="logs/{data_name}/linear/{gemini}_{mode}_run_{i}.err"
	shell:
		"python -u sparse_gemini/main.py --gemini {wildcards.gemini} --csv {input[0]} --mode {wildcards.mode} " + \
		"--output_path {params.output_dir} {params.affinity} -M 0 " + config["lasso_vars"] + " > {log.out} 2> {log.err}"

rule make_distance:
	input:
		config['data_path']+"/{data_name}_{i}.csv"
	output:
		config['distance_path']+"/{data_name}_{i}_{dist_type}.csv"
	params:
		get_kernel_option
	log:
		out="logs/{data_name}/preprocess/create_{dist_type}_{i}.out",
		err="logs/{data_name}/preprocess/create_{dist_type}_{i}.err"
	shell:
		"python -u utils/compute_distances.py {params} --dataset {input} --output {output} " + config["distance_vars"]+ \
		" > {log.out} 2> {log.err}"

rule make_dataset:
	input:
		get_source
	output:
		f"{config['data_path']}"+"/{data_name}_{i}.csv"
	log:
		out="logs/{data_name}/preprocess/create_dataset_{i}.out",
		err="logs/{data_name}/preprocess/create_dataset_{i}.err"
	shell:
		"python -u utils/"+config['data_script']+" --output_path {output} "+config['data_vars']+\
		" > {log.out} 2> {log.err}"

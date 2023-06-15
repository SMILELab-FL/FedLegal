"""Grid Search for FedETuning"""
import argparse
import os
import sys
import itertools as it
from loguru import logger
from multiprocessing import Pool
from configs.tuning import hyperparameter_grid


def run_process(proc):
    os.system(proc)

run_dirs = sys.argv[1]
task_name = sys.argv[2]
fl_algorithm = sys.argv[3]
config_path = sys.argv[4]
# partition_method = sys.argv[4]
tuning_type = sys.argv[5]
port_start = int(sys.argv[6])
device = sys.argv[7]

device_idx_list = [idx for idx in device.split(",")]
n_gpu = len(device_idx_list)
world_size = 1
logger.info(f"world_size is {world_size}")
logger.info(f"run_dirs is {run_dirs}")

max_seq = 512
dataset_name = "legal"
metric_name = "legal"
model_name = 'roberta-wwm-ext'
data_file = 'legal/silo'
if task_name == "LCP":
    model_output_mode = "seq_classification"
elif task_name == 'LJP':
    model_output_mode = 'seq_regression'
elif task_name == 'LER':
    model_output_mode = "token_classification_crf"
elif task_name == 'LRE':
    model_output_mode = "seq_classification"
elif task_name == 'LAM':
    model_output_mode = "multi_seq_classification"
elif task_name == 'LDG':
    model_output_mode = 'seq_generation'
    model_name = 'gpt2-chinese-cluecorpussmall'
    max_seq = 1024
else:
    logger.info(f"not support {task_name}")

logger.info(f"{task_name}'s max_seq is {max_seq}")

cmds = []
gpu_index = 0
hyper_parameter = hyperparameter_grid[tuning_type]
for parameter in it.product(*list(hyper_parameter.values())):
    specific_parameter_dict = {key: parameter[list(hyper_parameter.keys()).index(key)]
                               for key in list(hyper_parameter.keys())}
    if "lora_rank" in specific_parameter_dict:
        specific_parameter_dict["lora_alpha"] = specific_parameter_dict["lora_rank"]
    port = port_start + gpu_index
    device_index = gpu_index % n_gpu

    cmd = f'CUDA_VISIBLE_DEVICES={device_idx_list[device_index]} python main.py '
    options = [
        "--model_name_or_path", f"{run_dirs}/pretrain/nlp/{model_name}/", # roberta-wwm-ext
        "--output_dir", f"{run_dirs}/output/{data_file}",
        "--task_name", f"{task_name}",
        "--fl_algorithm", f"{fl_algorithm}",
        "--raw_dataset_path", f"{run_dirs}/datasets/{data_file}",
        "--partition_dataset_path", f"{run_dirs}/datasets/{data_file}",
        "--max_seq_length", f"{max_seq}",
        "--world_size", f"{world_size}",
        "--port", f"{port}",
        "--dataset_name", dataset_name,
        "--metric_name", metric_name,
        "--model_output_mode", model_output_mode,
        "--tuning_type", f"{tuning_type}_{model_name}",
        "--raw_tuning_type", tuning_type,
        "--config_path", config_path,
        "--do_grid", "True",
    ]
    for key, value in specific_parameter_dict.items():
        options.extend(["--" + key, str(value)])

    if fl_algorithm == 'centralized':
        cen_options = options + ["--rank", "-1"]
    elif fl_algorithm == 'local':
        cen_options = options + ["--rank", "-2"]
    cen_cmd = cmd + " ".join(cen_options)

    gpu_index += 1
    cmds.append(cen_cmd)

run_process("sleep 3s")
logger.warning(f"run {len(cmds)} grid-search tasks for {model_name}_{task_name}_{tuning_type}")

# run_process(cmds[0])  # debug
pool = Pool(processes=n_gpu)
pool.map(run_process, cmds)

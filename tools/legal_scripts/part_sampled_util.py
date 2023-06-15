import argparse
import random
import os
import sys
from loguru import logger
from copy import deepcopy

sys.path.append("../../")
from utils import pickle_write, pickle_read, make_sure_dirs, setup_seed

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, help="run directory of user machine")
    parser.add_argument("--task", type=str, required=True, help="Task name")
                        # LCP, LJP, LER, LRE, LAM
    parser.add_argument("--full_sampled_dir", default='/datasets/legal', type=str,
                        help="The directory to save partition or raw data under full sampling")
    parser.add_argument("--output_dir", default='/datasets/legal', type=str,
                        help="The output directory to save partition or raw data under sampling")
    parser.add_argument("--overwrite", default=False, type=bool,
                        help="overwrite")
    parser.add_argument("--seed", default=42, type=int,
                        help="seed")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info("start...")
    args = parser_args()
    setup_seed(args.seed)

    run_dir = args.run_dir
    args.output_dir = run_dir + args.output_dir
    args.full_sampled_dir = run_dir + args.full_sampled_dir

    # silo partition and heuristic partition
    # args.overwrite = False
    full_sampled_dir = os.path.join(args.full_sampled_dir, "silo")
    make_sure_dirs(full_sampled_dir)
    args.global_data_file = os.path.join(full_sampled_dir, f"{args.task.lower()}_data.pkl")
    args.partition_file = os.path.join(full_sampled_dir, f"{args.task.lower()}_partition.pkl")
    logger.info(f"full_sampled_dir: {full_sampled_dir}")
    logger.info(f"full sampled global_data_file: {args.global_data_file}")
    logger.info(f"full sampled partition_file: {args.partition_file}")

    global_obj, partition_obj = pickle_read(args.global_data_file), pickle_read(args.partition_file)
    sampled_partition = deepcopy(partition_obj)
    
    sample_ratio_list = [0.1, 0.5]
    for sample_ratio in sample_ratio_list:
        output_dir = os.path.join(args.output_dir, f"silo_sampled_{sample_ratio}/{args.task.lower()}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_train_data = []
        for client_idx, example_list in partition_obj['silo']['train'].items():
            print(f"origin client {client_idx} sample data size: {len(example_list)}")
            random.shuffle(example_list)
            sampled_partition['silo']['train'][client_idx] = example_list[: int(sample_ratio * len(example_list))]
            print(f"client {client_idx} sample data size: {len(sampled_partition['silo']['train'][client_idx])}")
    
            all_train_data = all_train_data + sampled_partition['silo']['train'][client_idx]
            # print(value.keys())
            # print(len(sampled_obj['global_train']), sampled_obj['global_train'][:1])

        sampled_file = f'{output_dir}/{args.task.lower()}_partition.pkl'
        if os.path.isfile(sampled_file) and not args.overwrite:
            logger.info(f"Partition method 'silo_partition' has existed for sample ratio={sample_ratio}, "
                        f"and overwrite={args.overwrite}, then skip")
        else:
            pickle_write(sampled_partition, sampled_file)

        global_file = f'{output_dir}/{args.task.lower()}_data.pkl'
        if os.path.isfile(global_file) and not args.overwrite:
            logger.info(f"Centalized data has existed for sample ratio={sample_ratio}, "
                        f"and overwrite={args.overwrite}, then skip")
        else:
            global_obj['global_train'] = all_train_data
            pickle_write(global_obj, global_file)
    
        print(f"writing {sample_ratio} successfully!")

""" general functions for FedETuning """

import os
import pickle
import importlib
import math
import multiprocessing

import random
import shutil
from datetime import time
from glob import glob

import numpy as np
import psutil
import torch


def pickle_read(path, read_format="rb"):
    with open(path, read_format) as file:
        obj = pickle.load(file)
    return obj


def pickle_write(obj, path, write_format="wb"):
    with open(path, write_format) as file:
        pickle.dump(obj, file)


def file_write(line, path, mode):
    with open(path, mode) as file:
        file.write(line + "\n")


def make_sure_dirs(path: str):
    """Create dir if not exists

    Args:
        path (str): path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def rm_dirs(path: str):
    """
    remove file existing check.
    Args:
        path (str): path
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def rm_file(file_path: str):
    try:
        if os.path.isfile(file_path) and os.path.exists(file_path):
            os.unlink(file_path)
    except:
        print(os.path.exists(file_path))


def get_cpus():
    """return total num of cpus in current machine."""
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            cfs_quota_us = int(f.readline())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            cfs_period_us = int(f.readline())
        if cfs_quota_us > 0 and cfs_period_us > 0:
            return int(math.ceil(cfs_quota_us / cfs_period_us))
    except Exception:
        pass
    return multiprocessing.cpu_count()


def get_memory_usage():
    """
        return total memory been used.
        memory use in GB
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2.0 ** 30
    return memory_use


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num / 1e6, 'Trainable': trainable_num / 1e6}


def check_cached_data(path):
    data_processed_flag_file = os.path.join(path, "server_write.flag")
    if not os.path.isfile(data_processed_flag_file):
        return False

    with open(data_processed_flag_file) as file:
        lines = []
        for line in file:
            lines.append(line)

        if not lines:
            return False

    return True


def setup_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    #  torch.backends.cudnn.benchmark = False
    #  torch.backends.cudnn.deterministic = True


def init_training_device(gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device


def global_metric_save(handler, training_config, logger):
    pickle_write(handler.metric_log, training_config.metric_log_file)
    handler.metric_line += f"global_valid_{handler.metric_to_check}={handler.global_valid_best_metric:.3f}_"
    handler.metric_line += f"global_test_{handler.metric_name}={handler.global_test_best_metric:.3f}"
    file_write(handler.metric_line, training_config.metric_file, "a+")
    logger.info(f"watch fed training logs --> {training_config.metric_log_file}")
    logger.info(f"global training metric --> {training_config.metric_file}")


def cen_metric_save(loc_trainer, training_config, logger):
    try:
        pickle_write(loc_trainer.metric_log, training_config.metric_log_file)
        logger.info(f"Rank_{loc_trainer.rank}, watch local training test logs --> {training_config.metric_log_file}")
    except:
        time.sleep(3)
        logger.warning(f"Rank_{loc_trainer.rank} loc_trainer {loc_trainer.metric_log}")

    # check_metric result record
    # global test
    valid_metric = loc_trainer.loc_best_metric[-1] if -1 in loc_trainer.loc_best_metric.keys() else 0
    line = training_config.metric_line + f"global_valid_{loc_trainer.metric_to_check}={valid_metric:.3f}_"
    test_metric = loc_trainer.loc_test_metric[-1] if -1 in loc_trainer.loc_test_metric.keys() else 0
    line += f"global_test_{loc_trainer.metric_name}={test_metric:.3f}"

    # local test
    test_metric = np.mean([value for idx, value in loc_trainer.loc_test_metric.items() if idx != -1])
    line += f"avg_local_test_{loc_trainer.metric_name}={test_metric:.3f}"
    file_write(line, training_config.metric_file, "a+")
    logger.info(f"result metric --> {training_config.metric_file}")


def local_metric_save(loc_trainer, training_config, logger):
    try:
        pickle_write(loc_trainer.metric_log, training_config.metric_log_file)
        logger.info(f"Rank_{loc_trainer.rank}, watch local training test logs --> {training_config.metric_log_file}")
    except:
        logger.warning(f"Rank_{loc_trainer.rank} loc_trainer {loc_trainer.metric_log}")

    valid_metric = np.mean(list(loc_trainer.loc_best_metric.values()))
    line = training_config.metric_line + f"avg_local_valid_{loc_trainer.metric_to_check}={valid_metric:.3f}_"
    test_metric = np.mean(list(loc_trainer.loc_test_metric.values()))
    line += f"avg_local_test_{loc_trainer.metric_name}={test_metric:.3f}"

    global_test_metric = np.mean(list(loc_trainer.global_test_metric.values()))
    line += f"avg_global_test_{loc_trainer.metric_name}={global_test_metric:.3f}"
    file_write(line, training_config.metric_file, "a+")
    logger.info(f"result metric --> {training_config.metric_file}")


def fed_local_metric_save(loc_trainer, training_config, logger):
    try:
        if loc_trainer.rank > 0 and len(loc_trainer.metric_log) > 0:  # federated: saving local test result
            pickle_write(loc_trainer.metric_log, training_config.metric_log_file)
            logger.info(
                f"Rank_{loc_trainer.rank}, watch local training test logs --> {training_config.metric_log_file}")
    except:
        logger.warning(f"Rank_{loc_trainer.rank}, loc")

    if len(loc_trainer.loc_test_metric) == 0:
        logger.warning("There is no local test metric info. Please check it!")
        return
    valid_metric, test_metric = 0, 0
    for idx in loc_trainer.loc_best_params:
        valid_metric += loc_trainer.loc_best_metric[idx]
        test_metric += loc_trainer.loc_test_metric[idx]
    valid_metric /= len(loc_trainer.loc_best_params)
    test_metric /= len(loc_trainer.loc_test_metric)

    line = training_config.metric_line + f"rank_{loc_trainer.rank}" + \
           f"local_valid_{loc_trainer.metric_to_check}={valid_metric:.3f}_"
    line += f"local_test_{loc_trainer.metric_name}={test_metric:.3f}"
    file_write(line, training_config.metric_file, "a+")
    logger.info(f"fed local training metric --> {training_config.metric_file}")


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': f"{total_num / 1e6:.3f}M", 'Trainable': f"{trainable_num / 1e6:.3f}M"}


def setup_imports():
    from utils.register import registry
    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that they register with registry
    root_folder = os.path.dirname(os.path.abspath(__file__))
    project_name = root_folder.split(os.sep)[-2]
    root_folder = os.path.join(root_folder, "..")  # check here
    files = []
    for package_name in ["trainers", "data", "models", "utils", "configs"]:
        folder = os.path.join(root_folder, package_name)
        pattern = os.path.join(folder, "**", "*.py")
        files.extend(glob(pattern, recursive=True))

    for f in files:
        f = os.path.realpath(f)
        if f.endswith(".py") and not f.endswith("__init__.py"):
            splits = f.split(os.sep)
            import_prefix_index = 0
            for idx, split in enumerate(splits):
                if split == project_name:
                    import_prefix_index = idx + 1
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(
                splits[import_prefix_index:-1] + [module_name]
            )
            importlib.import_module(module)

    registry.register("root_folder", root_folder)
    registry.register("imports_setup", True)

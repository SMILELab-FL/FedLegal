"""config for FedETuning"""

import os
import time
import copy
# import json
# import dataclasses
from abc import ABC
from omegaconf import OmegaConf
from transformers import HfArgumentParser

from utils import make_sure_dirs, rm_file
from utils.register import registry
from configs import ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments
from configs.tuning import get_delta_config, get_delta_key, hyperparameter_grid

grid_hyper_parameters = {}

for hyper_type, hyper_result in hyperparameter_grid.items():
    grid_hyper_parameters[hyper_type] = list(hyper_result.keys()) + ["tuning_type", "raw_tuning_type"]

# grid_hyper_parameters = ["tuning_type", "prefix_token_num", "prefix_token_num", "bottleneck_dim",
#                          "learning_rate", "dataset_name", "metric_name", "model_output_mode", "multi_label_threshold",
#                          "m_t", "v_t", "eta", "tau", "beta_1", "beta_2", "prox_mu"]


class Config(ABC):
    def __init__(self, model_args, data_args, training_args, federated_args):
        self.model_config = model_args
        self.data_config = data_args
        self.training_config = training_args
        self.federated_config = federated_args

    def save_configs(self):
        ...

    def check_config(self):
        self.config_check_federated()
        self.config_check_model()
        self.config_check_tuning()

    def config_check_federated(self):

        if "cen" in self.F.fl_algorithm:
            if self.F.rank == -1:
                self.F.world_size = 1
            else:
                raise ValueError(f"Please check Fl_algorithm and rank. Must set rank=-1, but find {self.F.rank}")
        elif "local" in self.F.fl_algorithm:
            if self.F.rank == -2:
                self.F.world_size = 1
            else:
                raise ValueError(f"Please check Fl_algorithm and rank. Must set rank=-2, but find {self.F.rank}")
        else:
            if self.F.clients_num % (self.F.world_size - 1):
                # raise ValueError(f"{self.F.clients_num} % {(self.F.world_size - 1)} != 0")
                pass

    def config_check_model(self):
        ...

    def config_check_tuning(self):

        if not self.M.tuning_type or "fine" in self.M.tuning_type:
            delta_config = {"delta_type": "fine-tuning"}
            self.M.tuning_type = ""
        else:
            delta_args = get_delta_config(self.M.tuning_type)
            if self.D.task_name in delta_args:
                delta_config = delta_args[self.D.task_name]
            else:
                delta_config = delta_args

        # TODO hard code for do grid search
        if self.T.do_grid:
            for key in delta_config:
                if getattr(self.M, key, None):
                    delta_config[key] = getattr(self.M, key)

                if key == "learning_rate" or key == "num_train_epochs":
                    delta_config[key] = getattr(self.T, key)

        registry.register("delta_config", delta_config)

        for config in [self.T, self.M, self.F, self.D]:
            for key, value in delta_config.items():
                if getattr(config, key, None) is not None:
                    setattr(config, key, value)
                    # registry.debug(f"{key}={value}")
        self.T.tuning_type = delta_config["delta_type"]
        # TODO hard code
        if "fed" in self.F.fl_algorithm and "lora" in self.T.tuning_type:
            self.T.num_train_epochs = 1
            delta_config["num_train_epochs"] = self.T.num_train_epochs

    @property
    def M(self):
        return self.model_config

    @property
    def D(self):
        return self.data_config

    @property
    def T(self):
        return self.training_config

    @property
    def F(self):
        return self.federated_config


def amend_config(model_args, data_args, training_args, federated_args):
    config = Config(model_args, data_args, training_args, federated_args)

    if config.F.rank > 0:
        # let server firstly start
        time.sleep(2)

    # set default metric
    if config.T.metric_for_best_model is None and config.T.greater_is_better is None:
        config.T.metric_for_best_model = 'loss'
        config.T.greater_is_better = False


    # load customer config (hard code)
    # TODO args in config.yaml can overwrite --arg
    root_folder = registry.get("root_folder")
    if not config.T.config_path and config.T.config_path != '':
        cust_config_path = os.path.join(root_folder, f"run/{config.F.fl_algorithm}/config.yaml")
    else:
        cust_config_path = os.path.join(root_folder, f"{config.T.config_path}")
        
    if os.path.isfile(cust_config_path):
        cust_config = OmegaConf.load(cust_config_path)
        for key, values in cust_config.items():
            if values:
                args = getattr(config, key)
                for k, v in values.items():
                    if config.T.do_grid and k in grid_hyper_parameters[config.M.raw_tuning_type]:
                        # grid search not overwrite --arg
                        continue
                    setattr(args, k, v)

    # set training path
    config.T.output_dir = os.path.join(config.T.output_dir, config.D.task_name)
    make_sure_dirs(config.T.output_dir)

    if not config.D.cache_dir:
        cache_dir = os.path.join(config.T.output_dir, "cached_data")
        if config.F.rank >= 0:
            config.D.cache_dir = os.path.join(
                cache_dir, f"cached_{config.M.model_type}_{config.F.partition_method}_{config.F.clients_num}_{config.F.alpha}"
            )
        else:
            config.D.cache_dir = os.path.join(
                cache_dir, f"cached_{config.M.model_type}_{config.F.partition_method}_{config.F.fl_algorithm}"
            )
    make_sure_dirs(config.D.cache_dir)

    # set training_args
    config.T.save_dir = os.path.join(config.T.output_dir, config.F.fl_algorithm.lower())
    make_sure_dirs(config.T.save_dir)
    config.T.checkpoint_dir = os.path.join(config.T.save_dir, "saved_model")
    make_sure_dirs(config.T.checkpoint_dir)

    # set phase
    phase = "train" if config.T.do_train else "evaluate"
    registry.register("phase", phase)

    # set metric log path
    times = time.strftime("%Y%m%d%H%M%S", time.localtime())
    registry.register("run_time", times)
    config.T.times = times
    config.T.metric_file = os.path.join(config.T.save_dir, f"{config.M.model_type}.eval")
    config.T.metric_log_file = os.path.join(config.T.save_dir, f"{times}_{config.M.model_type}.eval.log")
    if config.T.load_model_test:
        config.T.analysis = True
        config.T.metric_log_file = os.path.join(config.T.save_dir, f"{times}_{config.M.model_type}.eval.load_analysis.log")
    # distinguish different client rank
    if config.F.rank > 0:
        config.T.metric_log_file = os.path.join(config.T.save_dir, f"{times}_{config.M.model_type}_rank_{config.F.rank}.eval.log")
        if config.T.load_model_test:
            config.T.analysis = True
            config.T.metric_log_file = os.path.join(config.T.save_dir,
                                                    f"{times}_{config.M.model_type}_rank_{config.F.rank}.eval.load_analysis.log")

    # set federated_args
    if config.F.do_mimic and config.F.rank == 0:
        # wait for server processes data
        server_write_flag_path = os.path.join(config.D.cache_dir, "server_write.flag")
        rm_file(server_write_flag_path)

    if config.F.partition_method is None:
        config.F.partition_method = f"clients={config.F.clients_num}_alpha={config.F.alpha}"

    config.check_config()

    if config.T.do_grid:
        key_name, key_abb = get_delta_key(config.T.tuning_type)
        delta_config = registry.get("delta_config")
        if key_name:
            grid_info = "=".join([key_abb, str(delta_config[key_name])])
        else:
            grid_info = ""
        registry.register("grid_info", grid_info)

        config.T.metric_line = f"{times}_{config.M.model_type}_{config.T.tuning_type}_" \
                               f"cli={config.F.clients_num}_alp={config.F.alpha}_" \
                               f"sap={config.F.sample}_epo={config.T.num_train_epochs}_" \
                               f"lr={config.T.learning_rate}_{grid_info}_"
    else:
        config.T.metric_line = f"{times}_{config.M.model_type}_{config.T.tuning_type}_" \
                               f"cli={config.F.clients_num}_alp={config.F.alpha}_" \
                               f"sap={config.F.sample}_rd={config.F.rounds}_epo={config.T.num_train_epochs}_" \
                               f"lr={config.T.learning_rate}_"

    # addition hyper-param
    if config.F.fl_algorithm == 'fedprox' or config.F.fl_algorithm == 'ditto':
        config.T.metric_line += f'mu={config.F.prox_mu}_'

    if config.F.fed_opt_type is not None:  # fedopt
        config.T.metric_line += f"fed_opt_type={config.F.fed_opt_type}_m_t_{config.F.m_t}_v_t_{config.F.v_t}_" \
                                f"beta1_{config.F.beta_1}_beta2_{config.F.beta_2}_eta_{config.F.eta}_tau_{config.F.tau}_"

    if config.D.task_name == 'lam' and config.T.multi_label_threshold:
        config.T.metric_line += f"mlthred={config.T.multi_label_threshold}_"
    if config.D.task_name == 'ler' and config.T.crf_learning_rate:
        config.T.metric_line += f"crflr={config.T.crf_learning_rate}_"

    if config.training_config.multi_label_threshold:
        config.training_config.multi_label_threshold = list(config.training_config.multi_label_threshold)

    registry.register("config", config)

    return config


def build_config(args):
    # read parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments))
    model_args, data_args, training_args, federated_args = parser.parse_args_into_dataclasses(args=args)

    # amend and register configs
    config = amend_config(model_args, data_args, training_args, federated_args)
    delta_config = registry.get("delta_config")

    # logging fl & some path
    logger = registry.get("logger")
    logger.info(f"FL-Algorithm: {config.federated_config.fl_algorithm}")
    logger.info(f"output_dir: {config.training_config.output_dir}")
    logger.info(f"cache_dir: {config.data_config.cache_dir}")
    logger.info(f"save_dir: {config.training_config.save_dir}")
    logger.info(f"checkpoint_dir: {config.training_config.checkpoint_dir}")
    train_info = f"{config.M.model_type}_{delta_config['delta_type']}_" \
                 f"cli={config.F.clients_num}_alp={config.F.alpha}_rd={config.F.rounds}_sap={config.F.sample}_" \
                 f"lr={config.T.learning_rate}_epo={config.T.num_train_epochs}"

    # addition hyper-param
    if config.F.fl_algorithm == 'fedprox':
        config.T.metric_line += f'mu={config.F.prox_mu}_'

    if config.F.fed_opt_type is not None:  # fedopt
        config.T.metric_line += f"fed_opt_type={config.F.fed_opt_type}_m_t_{config.F.m_t}_v_t_{config.F.v_t}_" \
                                f"beta1_{config.F.beta_1}_beta2_{config.F.beta_2}_eta_{config.F.eta}_tau_{config.F.tau}_"

    if config.D.task_name == 'lam' and config.T.multi_label_threshold:
        config.T.metric_line += f"mlthred={config.T.multi_label_threshold}_"
    if config.D.task_name == 'ler' and config.T.crf_learning_rate:
        config.T.metric_line += f"crflr={config.T.crf_learning_rate}_"

    if config.training_config.multi_label_threshold:
        config.training_config.multi_label_threshold = list(config.training_config.multi_label_threshold)

    logger.debug(f"TrainBaseInfo: {train_info}")

    # logger.debug(delta_config)
    # exit()
    return config

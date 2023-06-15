"""Customer Dataloader for FedETuning"""

import os
import numpy as np

from utils import registry
from data import BaseDataLoader
from data.utils import conll_convert_examples_to_features, action_legal_examples_to_features

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import glue_convert_examples_to_features, AutoTokenizer, BertTokenizer


@registry.register_data("glue")
class GlueDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

        self.output_mode = self.attribute["output_mode"]
        self.label_list = self.attribute["label_list"]

        self._load_data()

    def _reader_examples(self, raw_data, partition_data, n_clients,
                         train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
                         train_features_dict, valid_features_dict, test_features_dict,
                         train_fedtures_all=None, valid_fedtures_all=None, test_fedtures_all=None
                         ):

        clients_partition_data = partition_data[self.partition_name]

        # TODO multi-task examples
        self.logger.info("convert train examples into features ...")
        train_features_all = np.array(glue_convert_examples_to_features(
            examples=raw_data["train"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert valid examples into features ...")
        valid_fedtures_all = np.array(glue_convert_examples_to_features(
            examples=raw_data["valid"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert test examples into features ...")
        test_fedtures_all = np.array(glue_convert_examples_to_features(
            examples=raw_data["test"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("build clients train & valid features ...")
        for idx in range(n_clients):
            client_train_list = clients_partition_data["train"][idx]
            train_examples_num_dict[idx] = len(client_train_list)
            train_features_dict[idx] = train_features_all[client_train_list]

            client_valid_list = clients_partition_data["valid"][idx]
            valid_examples_num_dict[idx] = len(client_valid_list)
            valid_features_dict[idx] = valid_fedtures_all[client_valid_list]

        self.train_num, self.valid_num, self.test_num = \
            len(train_features_all), len(valid_fedtures_all), len(test_fedtures_all)

        federated_data = (
            train_features_dict, valid_features_dict, test_features_dict,
            train_fedtures_all, valid_fedtures_all, test_fedtures_all,
            train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
            self.train_num, self.valid_num, self.test_num
        )

        return federated_data


@registry.register_data("ner")
class NERDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

        self.output_mode = self.attribute["output_mode"]
        self.label_list = self.attribute["label_list"]

        self._load_data()

    def _reader_examples(self, raw_data, partition_data, n_clients,
                         train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
                         train_features_dict, valid_features_dict, test_features_dict,
                         train_fedtures_all=None, valid_fedtures_all=None, test_fedtures_all=None
                         ):
        clients_partition_data = partition_data[self.partition_name]

        self.logger.info("convert train examples into features ...")
        train_features_all = np.array(conll_convert_examples_to_features(
            examples=raw_data["train"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert valid examples into features ...")
        valid_fedtures_all = np.array(conll_convert_examples_to_features(
            examples=raw_data["valid"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert test examples into features ...")
        test_fedtures_all = np.array(conll_convert_examples_to_features(
            examples=raw_data["test"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("build clients train & valid features ...")
        for idx in range(n_clients):
            client_train_list = clients_partition_data["train"][idx]
            train_examples_num_dict[idx] = len(client_train_list)
            train_features_dict[idx] = train_features_all[client_train_list]

            client_valid_list = clients_partition_data["valid"][idx]
            valid_examples_num_dict[idx] = len(client_valid_list)
            valid_features_dict[idx] = valid_fedtures_all[client_valid_list]

        self.train_num, self.valid_num, self.test_num = \
            len(train_features_all), len(valid_fedtures_all), len(test_fedtures_all)

        federated_data = (
            train_features_dict, valid_features_dict, test_features_dict,
            train_fedtures_all, valid_fedtures_all, test_fedtures_all,
            train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
            self.train_num, self.valid_num, self.test_num
        )

        return federated_data


@registry.register_data("legal")
class LegalDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

        self.output_mode = self.attribute["output_mode"]
        if 'crf' in self.model_config.model_output_mode:  # for crf process
            self.output_mode = self.model_config.model_output_mode
        self.label_list = self.attribute["label_list"]

        self._load_data()

    def _reader_examples(self, raw_data, partition_data, n_clients,
                         train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
                         train_features_dict, valid_features_dict, test_features_dict,
                         train_fedtures_all=None, valid_fedtures_all=None, test_fedtures_all=None
                         ):
        clients_partition_data = partition_data[self.partition_name]

        self.logger.info("convert train examples into features ...")
        train_fedtures_all = action_legal_examples_to_features(
            examples=raw_data["global_train"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        )

        self.logger.info("convert valid examples into features ...")
        valid_fedtures_all = action_legal_examples_to_features(
            examples=raw_data["global_valid"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        )

        self.logger.info("convert test examples into features ...")
        test_fedtures_all = action_legal_examples_to_features(
            examples=raw_data["global_test"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        )

        self.logger.info("build clients train & valid features ...")
        for idx in range(n_clients):            
            train_examples_num_dict[idx] = len(clients_partition_data["train"][idx])
            train_features_dict[idx] = action_legal_examples_to_features(
                examples=clients_partition_data["train"][idx], tokenizer=self.tokenizer,
                max_length=self.data_config.max_seq_length,
                label_list=self.label_list, output_mode=self.output_mode
            )

            valid_examples_num_dict[idx] = len(clients_partition_data["valid"][idx])
            valid_features_dict[idx] = action_legal_examples_to_features(
                examples=clients_partition_data["valid"][idx], tokenizer=self.tokenizer,
                max_length=self.data_config.max_seq_length,
                label_list=self.label_list, output_mode=self.output_mode
            )
            
            # test_examples_num_dict[idx] = len(client_valid_list)
            test_features_dict[idx] = action_legal_examples_to_features(
                examples=clients_partition_data["test"][idx], tokenizer=self.tokenizer,
                max_length=self.data_config.max_seq_length,
                label_list=self.label_list, output_mode=self.output_mode
            )

        self.train_num, self.valid_num, self.test_num = \
            len(train_fedtures_all), len(valid_fedtures_all), len(test_fedtures_all)

        federated_data = (
            train_features_dict, valid_features_dict, test_features_dict,
            train_fedtures_all, valid_fedtures_all, test_fedtures_all,
            train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
            self.train_num, self.valid_num, self.test_num
        )

        return federated_data

    def _build_tokenizer(self):
        if "gpt2" in self.model_config.model_type and "chinese" in self.model_config.model_type:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_config.model_name_or_path,
                # cache_dir=self.model_config.cache_dir,
                use_fast=True,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
                add_prefix_space=True,
                padding_side='left' if self.model_config.model_output_mode == 'seq_generation' else 'right'
            )
            # registry.register("tokenizer", self.tokenizer)
        elif self.model_config.model_type in {"bloom", "gpt2", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name_or_path,
                # cache_dir=self.model_config.cache_dir,
                use_fast=True,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
                add_prefix_space=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name_or_path,
                # cache_dir=self.model_config.cache_dir,
                use_fast=True,
                revision=self.model_config.model_revision,
                use_auth_token=True if self.model_config.use_auth_token else None,
            )
        registry.register("tokenizer", self.tokenizer)

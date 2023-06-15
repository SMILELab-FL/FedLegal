"""centralized Trainer"""
import os
import time
from abc import ABC

import torch

from trainers import BaseTrainer
from utils import registry, local_metric_save
from run.local.client import LocalClientTrainer, LocalClientTrainerCRF
from fedlab.utils.serialization import SerializationTool


@registry.register_fl_algorithm("local")
class LocalTrainer(BaseTrainer, ABC):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_client(self):
        if 'crf' in self.model_config.model_output_mode:
            self.client_trainer = LocalClientTrainerCRF(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.valid_dataloader_dict,
            )
        else:
            self.client_trainer = LocalClientTrainer(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.valid_dataloader_dict,
            )

    def on_client_end(self):
        if self.training_config.do_predict:
            local_eval_result_dict = self.client_trainer.test_on_client_locally()  # client local test
            global_eval_result_dict = self.client_trainer.local_train_test_on_client_globally()

            self.client_trainer.metric_log['predict_global'] = global_eval_result_dict
            self.client_trainer.metric_log['predict_local'] = local_eval_result_dict
            local_metric_save(self.client_trainer, self.training_config, self.logger)




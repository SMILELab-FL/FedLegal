"""centralized Trainer"""
import os
import time
from abc import ABC

import torch

from trainers import BaseTrainer
from utils import registry, cen_metric_save
from run.centralized.client import CenClientTrainer, CenClientTrainerCRF
from fedlab.utils.serialization import SerializationTool


@registry.register_fl_algorithm("centralized")
class CentralizedTrainer(BaseTrainer, ABC):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_client(self):
        if 'crf' in self.model_config.model_output_mode:
            self.client_trainer = CenClientTrainerCRF(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.test_dataloader_dict,
            )
        else:
            self.client_trainer = CenClientTrainer(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.test_dataloader_dict,
            )

    def on_client_end(self):
        if self.training_config.do_predict:
            global_eval_result_dict = self.client_trainer.test_on_client_locally()  # global test
            local_eval_result_dict = self.client_trainer.cen_train_test_on_client_locally()  # using global model to test locally

            self.client_trainer.metric_log['predict_global'] = global_eval_result_dict
            self.client_trainer.metric_log['predict_local'] = local_eval_result_dict
            cen_metric_save(self.client_trainer, self.training_config, self.logger)

        if self.training_config.model_save:
            times = registry.get("run_time")
            # glo_save_file = os.path.join(
            #     self.training_config.checkpoint_dir, f"{times}_{self.model_config.model_type}.pth")
            # torch.save(self.model.state_dict(), glo_save_file)

            self.model.backbone.save_pretrained(os.path.join(self.training_config.checkpoint_dir, times))
            self.logger.info(f"model saving --> {os.path.join(self.training_config.checkpoint_dir, times)}")

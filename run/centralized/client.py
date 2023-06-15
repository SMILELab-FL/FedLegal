"""Centralized Training for FedETuning"""

from abc import ABC
from collections import defaultdict

import numpy as np
from fedlab.utils import SerializationTool

from trainers.BaseClient import BaseClientTrainer, CRFClientTrainer


class CenClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, test_dataset):
        super().__init__(model, train_dataset, valid_dataset, test_dataset)

    def cen_train_test_on_client_locally(self):
        temp_central = False
        if -1 in self.loc_test_metric.keys():
            temp = self.loc_test_metric.pop(-1)  # delete global test result
            temp_central = True

        eval_result_dict = defaultdict(dict)
        for idx in range(self.federated_config.clients_num):
            test_dataloader = self._get_dataloader(dataset=self.test_dataset, client_id=idx)
            result = self.eval.test_and_eval(
                model=self._model,  # use global model
                valid_dl=test_dataloader,
                model_type=self.model_config.model_type,
                model_output_mode=self.model_config.model_output_mode
            )
            test_metric, test_loss = result[self.metric_name], result["eval_loss"]
            self.logger.critical(
                f"{self.data_config.task_name.upper()} Local Test, "
                f"Client {idx} , Local Test loss:{test_loss:.3f}, "
                f"Local Test {self.metric_name}:{test_metric:.3f}"
            )
            all_test_info = [f"Test {key}: {value:.3f}" for key, value in result.items()
                             if key not in self.metric.metric_log_skip_name]
            self.logger.critical(", ".join(all_test_info))

            self.loc_test_metric[idx] = test_metric
            eval_result_dict[idx] = result

        if len(self.loc_test_metric) > 0:  # skip no-local test
            self.logger.critical(
                f"#### Centralized training algorithm.  Local Test ####"
                f"Clients num: {len(self.loc_test_metric)}, Clients list: {list(self.loc_test_metric.keys())} \n"
                f"Centralized Avg Local Test {self.metric_name}:{np.mean(list(self.loc_test_metric.values())):.3f}, "
                f"Centralized All Local Test {self.metric_name}:{list(self.loc_test_metric.values())}"
            )
        if temp_central:
            self.loc_test_metric[-1] = temp

        return eval_result_dict


class CenClientTrainerCRF(CRFClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, test_dataset):
        super().__init__(model, train_dataset, valid_dataset, test_dataset)

    def cen_train_test_on_client_locally(self):
        temp_central = False
        if -1 in self.loc_test_metric.keys():
            temp = self.loc_test_metric.pop(-1)  # delete global test result
            temp_central = True

        eval_result_dict = defaultdict(dict)
        for idx in range(self.federated_config.clients_num):
            test_dataloader = self._get_dataloader(dataset=self.test_dataset, client_id=idx)
            result = self.eval.test_and_eval(
                model=self._model,  # use global model
                valid_dl=test_dataloader,
                model_type=self.model_config.model_type,
                model_output_mode=self.model_config.model_output_mode
            )
            test_metric, test_loss = result[self.metric_name], result["eval_loss"]
            self.logger.critical(
                f"{self.data_config.task_name.upper()} Local Test, "
                f"Client {idx} , Local Test loss:{test_loss:.3f}, "
                f"Local Test {self.metric_name}:{test_metric:.3f}"
            )
            all_test_info = [f"Test {key}: {value:.3f}" for key, value in result.items()
                             if key not in self.metric.metric_log_skip_name]
            self.logger.critical(", ".join(all_test_info))

            self.loc_test_metric[idx] = test_metric
            eval_result_dict[idx] = result

        if len(self.loc_test_metric) > 0:  # skip no-local test
            self.logger.critical(
                f"#### Centralized training algorithm.  Local Test ####"
                f"Clients num: {len(self.loc_test_metric)}, Clients list: {list(self.loc_test_metric.keys())} \n"
                f"Centralized Avg Local Test {self.metric_name}:{np.mean(list(self.loc_test_metric.values())):.3f}, "
                f"Centralized All Local Test {self.metric_name}:{list(self.loc_test_metric.values())}"
            )
        if temp_central:
            self.loc_test_metric[-1] = temp

        return eval_result_dict

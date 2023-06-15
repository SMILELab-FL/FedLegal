""" BaseServer for FedETuning """

import os
import random
import threading
from abc import ABC

import torch
import numpy as np

from utils.register import registry

from fedlab.core.server.handler import Aggregators
from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.core.server.manager import ServerManager
from fedlab.utils.serialization import SerializationTool
from fedlab.utils import MessageCode
from fedlab.core.coordinator import Coordinator


class BaseSyncServerHandler(ParameterServerBackendHandler, ABC):
    def __init__(self, model, valid_data, test_data):

        self.valid_data = valid_data
        self.test_data = test_data

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.logger = registry.get("logger")

        self.device = config.training_config.device
        self._model = model.to(self.device)

        # basic setting
        self.client_num_in_total = config.federated_config.clients_num
        self.sample_ratio = config.federated_config.sample

        # client buffer
        self.client_buffer_cache = []
        self.cache_cnt = 0

        # stop condition
        self.global_round = config.federated_config.rounds
        self.round = 0

        #  metrics & eval
        self._build_metric()
        self._build_eval()
        self.global_valid_best_metric = \
            -float("inf") if self.training_config.greater_is_better else float("inf")
        self.global_test_best_metric = 0.0
        self.metric_log = {
            "model_type": self.model_config.model_type,
            "clients_num": self.federated_config.clients_num,
            "alpha": self.federated_config.alpha, "task": self.data_config.task_name,
            "fl_algorithm": self.federated_config.fl_algorithm,
            "metric_for_best_model": self.training_config.metric_for_best_model,
            "info": f"{self.model_config.model_type}_{self.federated_config.fl_algorithm}_"
                    f"{self.federated_config.clients_num}_{self.federated_config.alpha}",
            "logs": [],
        }
        # metric line
        self.metric_name = self.metric.metric_name
        # Evaluate with param `metric_for_best_model`
        self.metric_to_check = self.training_config.metric_for_best_model
        if self.metric_to_check == "loss":  # .startswith("eval_"):
            self.metric_to_check = "eval_loss"

        times = registry.get("run_time")
        if self.training_config.do_grid:
            grid_info = registry.get("grid_info")
            self.metric_line = f"{times}_{self.model_config.model_type}_{self.training_config.tuning_type}_" \
                               f"rd={self.federated_config.rounds}_" \
                               f"cli={self.federated_config.clients_num}_alp={self.federated_config.alpha}_" \
                               f"sap={self.federated_config.sample}_epo={self.training_config.num_train_epochs}_" \
                               f"lr={self.training_config.learning_rate}_{grid_info}_"
        else:
            self.metric_line = f"{times}_{self.model_config.model_type}_{self.training_config.tuning_type}_" \
                               f"rd={self.federated_config.rounds}_" \
                               f"cli={self.federated_config.clients_num}_alp={self.federated_config.alpha}_" \
                               f"sap={self.federated_config.sample}_lr={self.training_config.learning_rate}_" \
                               f"epo={self.training_config.num_train_epochs}_"
        # global model
        self.glo_save_file = os.path.join(
            self.training_config.checkpoint_dir, f"{times}_{self.model_config.model_type}.pth")
        self.best_glo_params = None

    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(
            self.device, self.metric
        )

    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(
            self.data_config.task_name, self.training_config.greater_is_better
        )

    def stop_condition(self) -> bool:
        return self.round >= self.global_round

    def sample_clients(self):
        selection = random.sample(
            range(self.client_num_in_total),
            self.client_num_per_round
        )
        return selection

    def _update_global_model(self, payload):
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_parameters_list = self.client_buffer_cache
            self.logger.debug(
                f"Round {self.round+1} Finished. Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )

            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(model_parameters_list)
            SerializationTool.deserialize_model(self._model, serialized_parameters)
            self.round += 1

            if self.federated_config.eval_rounds and self.round % self.federated_config.log_eval_len == 0:
                self.valid_on_server()

            if self.federated_config.test_rounds:
                if self.round % self.federated_config.log_test_len == 0:
                    global_result = self.test_on_server()
                    self.metric_log[f"global_test_round_{self.round}"] = global_result

            # reset cache cnt
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))

    @property
    def downlink_package(self):
        """Property for manager layer. BaseServer manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def if_stop(self):
        """
        class:`NetworkManager` keeps monitoring this attribute,
        and it will stop all related processes and threads when ``True`` returned.
        """
        return self.round >= self.global_round

    def valid_on_server(self):

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=self.valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        self.on_round_end(result)

    def test_on_server(self):
        if self.best_glo_params is not None:
            temp_cparams = SerializationTool.serialize_model(self._model)
            SerializationTool.deserialize_model(self._model, self.best_glo_params)
        else:
            self.logger.warning("There is no best global params, please check it. We will take final model to test.")

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=self.test_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        self.logger.critical(f"task:{self.data_config.task_name}, Setting:{self.metric_log['info']}"
                             f"Global Test {self.metric_name}:{result[self.metric_name]:.3f}")

        all_test_info = [f"Global Test {key}: {value:.3f}" for key, value in result.items()
                         if key not in self.metric.metric_log_skip_name]
        self.logger.critical(", ".join(all_test_info))

        self.global_test_best_metric = result[self.metric_name]
        # return to current global model
        if self.best_glo_params is not None:
            SerializationTool.deserialize_model(self._model, temp_cparams)

        return result

    def save_global_model(self):
        torch.save(self.best_glo_params, self.glo_save_file)

    def on_round_end(self, result):
        test_metric, test_loss = result[self.metric_name], result["eval_loss"]

        metric_value = result[self.metric_to_check]

        operator = np.greater if self.training_config.greater_is_better else np.less
        if operator(metric_value, self.global_valid_best_metric):
            self.global_valid_best_metric = metric_value
            self.best_glo_params = SerializationTool.serialize_model(self._model)

        self.logger.info(f"{self.data_config.task_name}-{self.model_config.model_type} "
                         f"train with client={self.federated_config.clients_num}_"
                         f"alpha={self.federated_config.alpha}_"
                         f"epoch={self.training_config.num_train_epochs}_"
                         f"seed={self.training_config.seed}_"
                         f"comm_round={self.federated_config.rounds}")

        self.logger.debug(f"{self.federated_config.fl_algorithm} Eval "
                          f"Round:{self.round}, Loss:{test_loss:.3f}, "
                          f"Current {self.metric_name}:{test_metric:.3f}, "
                          f"Best metric {self.metric_to_check}:{self.global_valid_best_metric:.3f}")

        self.metric_log["logs"].append(
            {f"round_{self.round}": result  # log all metric
                 # {"loss": f"{test_loss:.3f}",
                 #  f"{self.metric.metric_name}": f"{test_metric:.3f}"}
             }
        )


class BaseServerManager(ServerManager):
    """Synchronous communication

    BaseServerManager.run()
    setup() main_loop() shut_down()

    """

    def __init__(self, network, handler):
        super(BaseServerManager, self).__init__(network, handler)

        self.logger = registry.get("logger")

    def setup(self):
        self._network.init_network_connection()

        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map, mode='GLOBAL')  # mode='GLOBAL'
        if self._handler is not None:
            self._handler.client_num_in_total = self._handler.federated_config.clients_num  # self.coordinator.total

    def main_loop(self):

        while self._handler.if_stop is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()

                if message_code == MessageCode.ParameterUpdate:
                    if self._handler._update_global_model(payload):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):

        self.logger.info("TrainClient activation procedure")
        clients_this_round = self._handler.sample_clients()
        # old version  send rank and client_id
        # rank_dict = self.coordinator.map_id_list(clients_this_round)
        # new version  balance rank and client id
        rank_dict = self.coordinator.balance_id_list(clients_this_round, self._network.world_size - 1)

        self.logger.info("TrainClient id list: {}".format(clients_this_round))
        self.logger.info(f"Balance Client ids: {rank_dict}")

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(
                content=[id_list] + downlink_package,
                message_code=MessageCode.ParameterUpdate,
                dst=rank
            )

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.client_num_in_total)
        rank_dict = self.coordinator.balance_id_list(client_list, self._network.world_size-1)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.Exit,
                               dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(
            src=self._network.world_size - 1
        )
        assert message_code == MessageCode.Exit

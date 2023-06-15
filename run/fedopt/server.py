"""federated average server"""

from abc import ABC

import torch

from fedlab.utils import Aggregators, SerializationTool
from trainers.BaseServer import BaseSyncServerHandler, BaseServerManager


class FedOptSyncServerHandler(BaseSyncServerHandler, ABC):
    def __init__(self, model, valid_data, test_data, fed_opt_type, **kwargs):
        super().__init__(model, valid_data, test_data)
        option_fedopt_type = ['fedadagrad', 'fedyogi', 'fedadam', 'fedmsgd']
        assert fed_opt_type in option_fedopt_type
        self.fed_opt_type = option_fedopt_type.index(fed_opt_type)

        self.m_t, self.v_t = None, None
        for key, value in kwargs.items():  # beta_1, beta_2, eta, tau (m_t, v_t)
            setattr(self, key, value)
        self.logger.warning(f"self.fed_opt_type: {self.fed_opt_type}")

        assert self.fed_opt_type == 3 and self.m_t is not None  # given m_t for fedmsgd

    def _update_global_model(self, payload):
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_delta_param_list = self.client_buffer_cache
            self.logger.debug(
                f"Round {self.round + 1} Finished. Model parameters aggregation, number of aggregation elements "
                f"{len(model_delta_param_list)}"
            )

            # use aggregator for param dw
            serialized_delta_parameters = Aggregators.fedavg_aggregate(model_delta_param_list)

            # m_t
            if self.fed_opt_type != 3:  # adjusting m_t
                if self.m_t is None:
                    self.m_t = torch.zeros_like(serialized_delta_parameters)
                self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * serialized_delta_parameters

            # v_t
            if self.v_t is None:
                self.v_t = torch.zeros_like(serialized_delta_parameters)

            # fedopt
            if self.fed_opt_type == 0:  # fedadagrad
                self.v_t = self.v_t + torch.square(serialized_delta_parameters)
            elif self.fed_opt_type == 1:  # fedyogi
                self.v_t = self.v_t - (1 - self.beta_2) * torch.square(serialized_delta_parameters) * \
                           torch.sign(self.v_t - torch.square(serialized_delta_parameters))
            elif self.fed_opt_type == 2:  # fedadam
                self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * torch.square(serialized_delta_parameters)
            elif self.fed_opt_type == 3:  # fedmsgd, for FedNLP paper
                self.v_t = self.m_t * self.v_t + (1 - self.m_t) * serialized_delta_parameters

            if self.fed_opt_type == 3:
                new_serialized_params = self.model_parameters - self.eta * self.v_t
            else:  # standard fedopt
                new_serialized_params = self.model_parameters + self.eta * self.m_t / (torch.sqrt(self.v_t) + self.tau)

            SerializationTool.deserialize_model(self._model, new_serialized_params)
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


class FedOptServerManager(BaseServerManager, ABC):
    def __init__(self, network, handler):

        super().__init__(network, handler)

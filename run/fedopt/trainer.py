"""federated average trainer"""
from abc import ABC

from utils import registry
from trainers.FedBaseTrainer import BaseTrainer
from run.fedopt.client import FedOptClientTrainer, FedOptClientTrainerCRF, FedOptClientManager
from run.fedopt.server import FedOptSyncServerHandler, FedOptServerManager


@registry.register_fl_algorithm("fedopt")
class FedOptTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_server(self):
        self.handler = FedOptSyncServerHandler(
            self.model, valid_data=self.data.valid_dataloader,
            test_data=self.data.test_dataloader,
            fed_opt_type=self.federated_config.fed_opt_type,
            beta_1=self.federated_config.beta_1,
            beta_2=self.federated_config.beta_2,
            eta=self.federated_config.eta,
            tau=self.federated_config.tau,
            m_t=self.federated_config.m_t,
            v_t=self.federated_config.v_t,
        )

        self.server_manger = FedOptServerManager(
            network=self.network,
            handler=self.handler,
        )

    def _build_client(self):

        if 'crf' in self.model_config.model_output_mode:
            self.client_trainer = FedOptClientTrainerCRF(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.test_dataloader_dict
                # data_slices=self.federated_config.clients_id_list,
            )
        else:
            self.client_trainer = FedOptClientTrainer(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.test_dataloader_dict
                # data_slices=self.federated_config.clients_id_list,
            )

        self.client_manager = FedOptClientManager(
            trainer=self.client_trainer,
            network=self.network
        )

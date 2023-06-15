"""federated average trainer"""
from abc import ABC

from utils import registry
from trainers.FedBaseTrainer import BaseTrainer
from run.fedprox.client import FedProxClientTrainer, FedProxClientTrainerCRF, FedProxClientManager
from run.fedprox.server import FedProxSyncServerHandler, FedProxServerManager


@registry.register_fl_algorithm("fedprox")
class FedProxTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_server(self):
        self.handler = FedProxSyncServerHandler(
            self.model, valid_data=self.data.valid_dataloader,
            test_data=self.data.test_dataloader
        )

        self.server_manger = FedProxServerManager(
            network=self.network,
            handler=self.handler,
        )

    def _build_client(self):

        if 'crf' in self.model_config.model_output_mode:
            self.client_trainer = FedProxClientTrainerCRF(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.test_dataloader_dict
                # data_slices=self.federated_config.clients_id_list,
            )
        else:
            self.client_trainer = FedProxClientTrainer(
                model=self.model,
                train_dataset=self.data.train_dataloader_dict,
                valid_dataset=self.data.valid_dataloader_dict,
                test_dataset=self.data.test_dataloader_dict
                # data_slices=self.federated_config.clients_id_list,
            )

        self.client_manager = FedProxClientManager(
            trainer=self.client_trainer,
            network=self.network
        )

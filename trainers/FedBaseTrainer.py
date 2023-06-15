"""BaseTrainer for FedETuning"""
import os
from abc import ABC
from utils import registry
from utils import setup_seed
from utils import global_metric_save
from fedlab.core.network import DistNetwork
from utils import fed_local_metric_save


class BaseTrainer(ABC):
    def __init__(self, *args):

        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F

        self.logger = registry.get("logger")

        # self._before_training()
        self.edge_cases = None  # record in test phase

    @property
    def role(self):
        if self.federated_config.rank == 0:
            return "server"
        elif self.federated_config.rank > 0:
            return f"sub-server_{self.federated_config.rank}"
        else:
            return "centralized"

    def _build_server(self):
        raise NotImplementedError

    def _build_client(self):
        raise NotImplementedError

    def _build_local_trainer(self, *args):
        raise NotImplementedError

    def _build_network(self):
        self.network = DistNetwork(
            address=(self.federated_config.ip, self.federated_config.port),
            world_size=self.federated_config.world_size,
            rank=self.federated_config.rank,
            ethernet=self.federated_config.ethernet)

    def _build_data(self):
        self.data = registry.get_data_class(self.data_config.dataset_name)()

    def _build_model(self):
        self.model = registry.get_model_class(self.model_config.model_output_mode)(
            task_name=self.data_config.task_name
        )

    def _before_training(self):

        self.logger.info(f"{self.role} set seed {self.training_config.seed}")
        setup_seed(self.training_config.seed)

        self.logger.info(f"{self.role} building dataset ...")
        # set before build model
        self._build_data()

        self.logger.info(f"{self.role} building model ...")
        self._build_model()

        # self.logger.info(f"{self.role} building local trainer ...")
        # self._build_local_trainer()

        if self.federated_config.rank >= 0:
            self.logger.info(f"{self.role} building network ...")
            self._build_network()

        if self.federated_config.rank == 0:
            self.logger.info("building server ...")
            self._build_server()
        else:
            self._build_client()
            if self.federated_config.rank > 0:
                self.logger.info(f"building client {self.federated_config.rank} ...")
                self.logger.info(f"local rank {self.federated_config.rank}'s client ids "
                                 f"is {list(self.data.train_dataloader_dict.keys())}")
            elif self.federated_config.rank == -1:
                self.logger.info("building centralized training")
            elif self.federated_config.rank == -2:
                self.logger.info("building local training")

    def train(self):
        # TODO phase decides train or test
        if self.federated_config.rank == 0:
            self.logger.debug(f"Server Start ...")
            self.server_manger.run()
            self.on_server_end()

        elif self.federated_config.rank > 0:
            self.logger.debug(f"Sub-Server {self.federated_config.rank} Training Start ...")
            self.client_manager.run()
            self.on_client_end()

        elif 'cen' in self.federated_config.fl_algorithm:
            self.logger.debug(f"Centralized Training Start ...")
            self.client_trainer.cen_train()
            self.on_client_end()

        else:  # local
            self.logger.debug(f"Local Training Start ...")
            self.client_trainer.loc_train()
            self.on_client_end()


    def on_server_end(self):
        """on_server_end"""
        global_eval_result_dict = self.handler.test_on_server()
        self.handler.metric_log['predict_global'] = global_eval_result_dict
        global_metric_save(self.handler, self.training_config, self.logger)

        if self.training_config.model_save:
            times = registry.get("run_time")
            self.model.backbone.save_pretrained(os.path.join(self.training_config.checkpoint_dir, times))
            self.logger.info(
                f"global model saving --> {os.path.join(self.training_config.checkpoint_dir, f'{times}_global')}")

    def on_client_end(self, *args):
        """client local test Finally"""
        if self.training_config.do_predict:
            local_eval_result_dict = self.client_trainer.test_on_client_locally()  # TODO: 每个rank都有自己对best_param，所以存在重复test
            self.client_trainer.metric_log['predict_local'] = local_eval_result_dict
            fed_local_metric_save(self.client_trainer, self.training_config, self.logger)

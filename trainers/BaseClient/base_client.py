"""BaseClientTrainer for FedETuning"""

from abc import ABC
from typing import List
from thop import profile
from thop import clever_format

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW

from utils import registry
from utils import get_parameter_number
from fedlab.utils import MessageCode, SerializationTool
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.manager import ORDINARY_TRAINER, SERIAL_TRAINER


class BaseClientTrainer(ClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, test_dataset):

        self._model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self._before_training()

    def _before_training(self):
        """before training function"""

        self.type = SERIAL_TRAINER  # represent serial trainer

        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F

        self.client_num = len(config.F.clients_id_list)
        self.device = config.training_config.device
        self.rank = config.federated_config.rank
        self.param_list = []
        self.logger = registry.get("logger")

        self._build_metric()
        self._build_eval()

        # key: client idx, value: valid metric
        self.loc_best_metric = {}
        # key: client idx, value: test metric
        self.loc_test_metric = {}
        # key: client idx, value: serialized params
        self.loc_best_params = {}
        self.loc_cur_params = {}  # local model
        self.metric_log = defaultdict(dict)  # round eval metrics

        # local patient times and local early stop
        self.loc_patient_times = defaultdict(int)
        self.stop_early = defaultdict(bool)

        for idx in range(config.F.clients_num_in_total):
            self.loc_cur_params[idx] = SerializationTool.serialize_model(self._model)
        self.round = 0

        self.metric_name = self.metric.metric_name
        # Evaluate with param `metric_for_best_model`
        # if self.training_config.load_best_model_at_end:  # simpled param setting
        self.metric_to_check = self.training_config.metric_for_best_model
        if self.metric_to_check == "loss":  # .startswith("eval_"):
            self.metric_to_check = "eval_loss"

        self._model.to(self.device)

        if self.federated_config.rank == -1:
            self._calculate_model_computation()

    def _calculate_model_computation(self):

        dummy_idx = list(self.train_dataset.keys())[0]
        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=dummy_idx)
        for step, batch in enumerate(train_loader):
            self._model.train()
            batch = tuple(t.to(self.device) for t in batch)

            macs, params = profile(self._model.backbone, inputs=(batch[0],))
            flops, params = clever_format([macs, params], "%.3f")
            self.logger.debug(f"Model Type: {self.model_config.model_type}, "
                              f"Tuning Type: {self.training_config.tuning_type}, "
                              f"Parameters: {get_parameter_number(self._model.backbone)}, "
                              f"FLOPs: {flops}")
            break

    @property
    def uplink_package(self):
        return self.param_list

    def _train_alone(self, idx: int, model_parameters: torch.Tensor, *args, **kwargs):
        """local training for Client"""

        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
        if model_parameters is not None:
            SerializationTool.deserialize_model(self._model, model_parameters)

        # build optimizer,scheduler,loss
        optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
        self._model, optimizer = self._mixed_train_model(self._model, optimizer)
        self._build_loss()

        for epoch in range(0, int(self.training_config.num_train_epochs)):
            if self.rank < 0:
                self.round = epoch + 1  # simulate to control eval frequency in training
            self._on_epoch_begin()
            self._on_epoch(train_loader, optimizer, scheduler)
            self._on_epoch_end(idx)
            if self.federated_config.pson and self.stop_early[idx]:  # only useful for multi epochs
                self.logger.critical(f"local stop early in {epoch}")
                break

    def _get_dataloader(self, dataset, client_id: int):
        """Get :class:`DataLoader` for ``client_id``."""
        if isinstance(dataset, dict):
            data_loader = dataset[client_id]
        else:
            data_loader = dataset
        return data_loader

    def local_process(self, id_list: List, payload: List):
        """local process for Federated Learning"""
        self.round += 1
        model_parameters = payload[0]
        self.param_list = self.fed_train(model_parameters, id_list)
        return self.param_list

    def fed_train(self, model_parameters: torch.Tensor, id_list: List):
        param_list = []
        self.logger.info(f"Trainer id_list: {id_list}")
        self.logger.info(f"model_parameters: {model_parameters.shape}")
        self.logger.info(f"id_list: {id_list}")
        for idx in id_list:
            self._train_alone(
                idx=idx,
                model_parameters=model_parameters
            )
            param_list.append(self.model_parameters)

        return param_list

    def cen_train(self, *args):
        self._train_alone(
            idx=-1,
            model_parameters=None,
        )

    def loc_train(self, *args):
        for idx in range(self.client_num):
            self._train_alone(
                idx=idx,
                model_parameters=self.loc_cur_params[idx],
            )
            print(f"###### client {idx} save local model")
            self.loc_cur_params[idx] = SerializationTool.serialize_model(self._model)

    # Local Training Functions
    def _build_loss(self):
        self.criterion = registry.get_loss_class(self.training_config.loss_name)(
            config=self.training_config
        )

    def _build_optimizer(self, model, train_dl_len):
        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = self.get_optimized_model_params(model)

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler

    def get_optimized_model_params(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        # Both pieces of code have the same effect
        # optimizer_grouped_parameters = [
        #     {"params": filter(lambda x: x.requires_grad, model.bert.parameters()),
        #      'weight_decay': 0.0},
        # ]

        return optimizer_grouped_parameters

    def _mixed_train_model(self, model, optimizer):
        if self.training_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.training_config.fp16_opt_level)

            # multi-gpu training (should be after apex fp16 initialization)
        if self.training_config.n_gpu > 1:
            self.logger.warning("We haven't tested our model under multi-gpu. Please be aware!")
            model = torch.nn.DataParallel(model)

        return model, optimizer

    # Local Test Function
    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(
            self.data_config.task_name, self.training_config.greater_is_better
        )

    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(
            self.device, self.metric
        )

    def test_on_client_locally(self, assigned_idx=None):
        if len(self.loc_best_params) == 0:
            if self.federated_config.rank == -1:  # centralized
                self.logger.warning("There is no best global params currently. We will take the final global model "
                                    "to test for this centralized algorithm.")
                self.loc_best_params[-1] = SerializationTool.serialize_model(self._model)
            elif self.federated_config.rank == -2:  # local
                self.logger.warning("There is no best global params for each client. "
                                    "We will take the final local model of each client "
                                    "to test for this local algorithm.")
                for i, params in self.loc_cur_params.items():
                    self.loc_best_params[i] = params
            else:
                self.logger.warning("There is no best global params for each client currently. Please check it.")
                return None

        eval_result_dict = defaultdict(dict)
        if assigned_idx is None:
            test_idx = self.loc_best_params.keys()
        else:
            test_idx = [assigned_idx]
            temp_cparams = SerializationTool.serialize_model(self._model)
        self.logger.info(f"test_on_client_locally: {test_idx}")
        for idx in test_idx:
            loc_best_params = self.loc_best_params[idx]
            SerializationTool.deserialize_model(self._model, loc_best_params)
            test_dataloader = self._get_dataloader(dataset=self.test_dataset, client_id=idx)
            result = self.eval.test_and_eval(
                model=self._model,
                valid_dl=test_dataloader,
                model_type=self.model_config.model_type,
                model_output_mode=self.model_config.model_output_mode
            )
            test_metric, test_loss = result[self.metric_name], result["eval_loss"]
            self.logger.critical(
                f"{self.data_config.task_name.upper()} Local Test, "
                f"Client:{idx}, Local Test loss:{test_loss:.3f}, "
                f"Local Test {self.metric_name}:{test_metric:.3f}"
            )
            all_test_info = [f"Client:{idx}, Local Test {key}: {value:.3f}" for key, value in result.items()
                             if key not in self.metric.metric_log_skip_name]
            self.logger.critical(", ".join(all_test_info))

            self.loc_test_metric[idx] = test_metric

            eval_result_dict[idx] = result

        if assigned_idx is None:  # all clients, log avg result
            self.logger.critical(
                f"Clients num: {len(self.loc_test_metric)}, Clients list: {list(self.loc_test_metric.keys())} \n"
                f"Avg Test {self.metric_name}:{np.mean(list(self.loc_test_metric.values())):.3f}, "
                f"All Test {self.metric_name}:{list(self.loc_test_metric.values())}"
            )
        else:  # return to current client training result
            SerializationTool.deserialize_model(self._model, temp_cparams)
        return eval_result_dict

    # Local Epoch Function
    def _on_epoch_begin(self):
        self.global_step = 0
        self.tr_loss, self.logging_loss = 0.0, 0.0
        self.total, self.correct = 0, 0

    def _on_epoch(self, train_loader, optimizer, scheduler):
        with tqdm(total=len(train_loader), desc=f"Train rank {self.rank}") as pbar:
            for step, batch in enumerate(train_loader):
                # if step >= 2:  # debug
                #     break
                self._model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]
                          }
                label = inputs['labels']
                if self.model_config.model_type != 'distilbert' or self.model_config.model_type != 'roberta':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = batch[2] \
                        if self.model_config.model_type in ['bert', 'xlnet'] else None
                outputs = self._model(inputs)

                loss, logits = outputs[:2]
                _, predicted = torch.max(logits, 1)

                optimizer.zero_grad()
                if self.training_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.training_config.gradient_accumulation_steps > 1:
                    loss = loss / self.training_config.gradient_accumulation_steps

                # print(f"loss: {loss}")
                if self.training_config.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.tr_loss += loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})
                pbar.update(1)

                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.training_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.training_config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule

                    self.global_step += 1

                # calculate training results
                self._train_pred_info(logits, predicted, label)

    def _on_epoch_end(self, idx):
        """on epoch end"""

        self.logger.info(f"{self.data_config.task_name.upper()} Train, "
                         f"Client:{idx}, Loss:{self.tr_loss / self.global_step:.3f}, "
                         f"Accuracy:{self.correct / self.total:.3f}")

        if not self.federated_config.pson or self.round % self.federated_config.pson_round_len != 0:
            # not need for local test
            return

        valid_data = self._get_dataloader(dataset=self.valid_dataset, client_id=idx)

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        metric_value = result[self.metric_to_check]

        operator = np.greater if self.training_config.greater_is_better else np.less
        if (
                self.loc_best_metric.get(idx, None) is None
                or operator(metric_value, self.loc_best_metric[idx])
        ):
            self.loc_best_metric[idx] = metric_value
            self.loc_best_params[idx] = SerializationTool.serialize_model(self._model)
            self.loc_patient_times[idx] = 0
        else:
            self.loc_patient_times[idx] += 1

        self.logger.debug(f"{self.data_config.task_name.upper()} Eval, "
                          f"Client:{idx} epoch end, Local Eval Loss:{result['eval_loss']:.3f}, "
                          f"Current {self.metric_name}:{result[self.metric_name]:.3f}, "
                          f"Best metric {self.metric_to_check}:{self.loc_best_metric[idx]:.3f}")
        if self.loc_patient_times[idx] >= self.training_config.patient_times:
            self.stop_early[idx] = True

        if self.round % self.federated_config.pson_log_test_len == 0:  # personalized test log, only use for federated
            eval_result_dict = self.test_on_client_locally(idx)  # local test for this client
            metric_title = f"ep_{self.round}" if self.rank < 0 else f"round_{self.round}"
            self.metric_log[metric_title][f"client_{idx}"] = eval_result_dict[idx]

    def _train_pred_info(self, logits, predicted, label):
        if "seq_classification" in self.model_config.model_output_mode:
            if "multi" in self.model_config.model_output_mode:
                predicted = torch.sigmoid(logits)
                predicted = predicted >= torch.tensor(self.training_config.multi_label_threshold).to(self.device)
                self.total += label.size(0) * label.size(1)
                self.correct += (predicted == label).sum().item()
            else:
                self.total += label.size(0)
                self.correct += (predicted == label).sum().item()
        elif self.model_config.model_output_mode == "token_classification":
            # all tokens test
            self.total += label.size(0) * label.size(1)
            _, predicted = torch.max(logits, 2)
            self.correct += (predicted == label).sum().item()
        elif self.model_config.model_output_mode == "seq_regression":
            self.total += label.size(0)
            self.correct += (predicted == label).sum().item()  # exact match
        elif self.model_config.model_output_mode == 'seq_generation':
            # predicted = np.squeeze(preds)
            self.total += label.size(0)
            self.correct = 0  # only focus on Pearson score


class BaseClientManager(PassiveClientManager, ABC):
    def __init__(self, network, trainer):
        self.logger = registry.get("logger")
        super().__init__(network, trainer, self.logger)

    def main_loop(self):
        """Actions to perform when receiving a new message, including local trainers.

        Main procedure of each client:
            1. client waits for data from server (PASSIVELY).
            2. after receiving data, client start local model trainers procedure.
            3. client synchronizes with server actively.
        """
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:

                id_list, payload = payload[0].to(
                    torch.int32).tolist(), payload[1:]

                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:  # serial
                    self._trainer.local_process(
                        id_list=id_list,
                        payload=payload
                    )

                elif self._trainer.type == ORDINARY_TRAINER:  # ordinary
                    assert len(id_list) == 1
                    self._trainer.local_process(payload=payload)

                self.synchronize()

            else:
                raise ValueError(f"Invalid MessageCode {message_code}. Please check MessageCode list.")

    def synchronize(self):
        """Synchronize with server"""
        self.logger.info("Uploading information to server.")
        self._network.send(
            content=self._trainer.uplink_package,
            message_code=MessageCode.ParameterUpdate,
            dst=0
        )

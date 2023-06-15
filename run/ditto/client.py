"""federated average client"""

from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
import torch
from fedlab.utils import SerializationTool
from tqdm import tqdm

from trainers.BaseClient import BaseClientTrainer, BaseClientManager


class DittoClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, test_dataset):
        super().__init__(model, train_dataset, valid_dataset, test_dataset)
        self.loc_cur_params = {}
        for idx in range((self.rank - 1) * self.federated_config.clients_num_per_sub_server,
                         self.rank * self.federated_config.clients_num_per_sub_server):
            self.loc_cur_params[idx] = SerializationTool.serialize_model(self._model)
        self.logger.warning(f"rank :{self.rank}, loc_cur_params keys {self.loc_cur_params.keys()}")

    def fed_train(self, model_parameters: torch.Tensor, id_list: List):
        param_list = []
        self.logger.info(f"Trainer id_list: {id_list}")
        self.logger.info(f"model_parameters: {model_parameters.shape}")
        self.logger.info(f"id_list: {id_list}")
        for idx in id_list:
            update_glb_model_params = self._train_alone(
                idx=idx,
                model_parameters=model_parameters
            )
            param_list.append(update_glb_model_params)

        return param_list

    def _train_alone(self, idx: int, model_parameters: torch.Tensor, *args, **kwargs):
        """local training for Client"""

        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
        if model_parameters is not None:  # global model
            SerializationTool.deserialize_model(self._model, model_parameters)

        # build optimizer,scheduler,loss


        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()

            optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
            self._model, optimizer = self._mixed_train_model(self._model, optimizer)
            self._build_loss()
            update_glb_model_params = self._on_epoch_global(train_loader, optimizer, scheduler)

            # return to origin global model
            SerializationTool.deserialize_model(self._model, model_parameters)
            optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
            self._model, optimizer = self._mixed_train_model(self._model, optimizer)
            self._build_loss()
            update_local_model_params = self._on_epoch_local(train_loader, optimizer, scheduler, model_parameters, idx)

            self._on_epoch_end(idx)
            if self.federated_config.pson and self.stop_early:
                self.logger.critical(f"local stop early in {epoch}")
                break
        return update_glb_model_params

    def _on_epoch_global(self, train_loader, optimizer, scheduler):
        with tqdm(total=len(train_loader), desc=f"Train rank {self.rank} global model") as pbar:
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

                # self.tr_loss += loss.item()  # for local model train
                pbar.set_postfix({'Global model train loss': '{0:1.5f}'.format(loss)})
                pbar.update(1)

                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.training_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.training_config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule

                    # self.global_step += 1

                # calculate training results  # for local model train
                self._train_pred_info(logits, predicted, label)

        return self.model_parameters  # global model update

        # load local model to train
    def _on_epoch_local(self, train_loader, optimizer, scheduler, global_model_parameters, idx):
        SerializationTool.deserialize_model(self._model, global_model_parameters)
        frz_model_params = [deepcopy(param) for param in self._model.parameters()]

        SerializationTool.deserialize_model(self._model, self.loc_cur_params[idx])
        with tqdm(total=len(train_loader), desc=f"Train rank {self.rank} local model") as pbar:
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

                loss_origin, logits = outputs[:2]
                _, predicted = torch.max(logits, 1)

                # ditto loss
                prox_loss = 0
                for w0, w in zip(frz_model_params, self._model.parameters()):
                    prox_loss += torch.sum(torch.pow(w - w0, 2))
                loss = loss_origin + 0.5 * self.federated_config.prox_mu * prox_loss

                optimizer.zero_grad()
                if self.training_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.training_config.gradient_accumulation_steps > 1:
                    loss = loss / self.training_config.gradient_accumulation_steps

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
                pbar.set_postfix({'Local model train loss': '{0:1.5f}'.format(loss)})
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

        self.loc_cur_params[idx] = self.model_parameters
        return self.model_parameters

    def _on_epoch_end(self, idx):
        """on epoch end"""

        self.logger.info(f"{self.data_config.task_name.upper()} Train, "
                         f"Client:{idx}, Loss:{self.tr_loss / self.global_step:.3f}, "
                         f"Accuracy:{self.correct / self.total:.3f}")

        if not self.federated_config.pson or self.round % self.federated_config.pson_round_len != 0:
            # not need for local test
            return

        valid_data = self._get_dataloader(dataset=self.valid_dataset, client_id=idx)

        # in case not set to local current model
        SerializationTool.deserialize_model(self._model, self.loc_cur_params[idx])
        result = self.eval.test_and_eval(
            model=self._model,  # for ditto, use trained local model
            valid_dl=valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        metric_value = result[self.metric_to_check]
        self.logger.warning(f"metric value for epoch end, {metric_value}")
        operator = np.greater if self.training_config.greater_is_better else np.less
        if (
                self.loc_best_metric.get(idx, None) is None
                or operator(metric_value, self.loc_best_metric[idx])
        ):
            self.loc_best_metric[idx] = metric_value
            self.loc_best_params[idx] = SerializationTool.serialize_model(self._model)
            self.loc_patient_times = 0
            self.logger.warning(f"loc best metric update, {self.loc_best_metric[idx]}")
        else:
            self.loc_patient_times += 1

        self.logger.debug(f"{self.data_config.task_name.upper()} Eval, "
                          f"Client:{idx} epoch end, Local Eval Loss:{result['eval_loss']:.3f}, "
                          f"Current {self.metric_name}:{result[self.metric_name]:.3f}, "
                          f"Best metric {self.metric_to_check}:{self.loc_best_metric[idx]:.3f}")
        # hard-code: to print all metrics for generation task
        if self.model_config.model_output_mode == 'seq_generation':
            log_info = f"Client {idx}, Eval "
            for key, value in result.items():
                log_info += f"metric {key}: {value:.3f};"
            self.logger.debug(log_info)

        if self.loc_patient_times >= self.training_config.patient_times:
            self.stop_early = True

        if self.rank > 0 and self.round % self.federated_config.pson_log_test_len == 0:  # personalized test log, only use for federated
            eval_result_dict = self.test_on_client_locally(idx)  # local test for this client
            metric_title = f"ep_{self.round}" if self.rank < 0 else f"round_{self.round}"
            self.metric_log[metric_title][f"client {idx}"] = eval_result_dict[idx]

    def test_on_client_locally(self, assigned_idx=None):
        if not self.federated_config.pson and len(self.loc_best_params) == 0:
            self.logger.warning("There is no best global params for each client. "
                                "We will take the final local model of each client "
                                "to test for this local algorithm.")
            for i, params in self.loc_cur_params.items():
                self.loc_best_params[i] = params

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


class DittoClientTrainerCRF(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, test_dataset):
        super().__init__(model, train_dataset, valid_dataset, test_dataset)
        self.loc_cur_params = {}
        for idx in range((self.rank - 1) * self.federated_config.clients_num_per_sub_server,
                         self.rank * self.federated_config.clients_num_per_sub_server):
            self.loc_cur_params[idx] = SerializationTool.serialize_model(self._model)
        self.logger.warning(f"rank :{self.rank}, loc_cur_params keys {self.loc_cur_params.keys()}")

    def fed_train(self, model_parameters: torch.Tensor, id_list: List):
        param_list = []
        self.logger.info(f"Trainer id_list: {id_list}")
        self.logger.info(f"model_parameters: {model_parameters.shape}")
        self.logger.info(f"id_list: {id_list}")
        for idx in id_list:
            update_glb_model_params = self._train_alone(
                idx=idx,
                model_parameters=model_parameters
            )
            param_list.append(update_glb_model_params)

        return param_list

    def _train_alone(self, idx: int, model_parameters: torch.Tensor, *args, **kwargs):
        """local training for Client"""

        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
        if model_parameters is not None:  # global model
            SerializationTool.deserialize_model(self._model, model_parameters)

        # build optimizer,scheduler,loss


        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()

            optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
            self._model, optimizer = self._mixed_train_model(self._model, optimizer)
            self._build_loss()
            update_glb_model_params = self._on_epoch_global(train_loader, optimizer, scheduler)

            # return to origin global model
            SerializationTool.deserialize_model(self._model, model_parameters)
            optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
            self._model, optimizer = self._mixed_train_model(self._model, optimizer)
            self._build_loss()
            update_local_model_params = self._on_epoch_local(train_loader, optimizer, scheduler, model_parameters, idx)

            self._on_epoch_end(idx)
            if self.federated_config.pson and self.stop_early:
                self.logger.critical(f"local stop early in {epoch}")
                break
        return update_glb_model_params

    def _on_epoch_global(self, train_loader, optimizer, scheduler):
        with tqdm(total=len(train_loader), desc=f"Train rank {self.rank} global model") as pbar:
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
                pred_tags = self.model.backbone.crf.decode(logits, inputs['attention_mask'])

                optimizer.zero_grad()
                if self.training_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.training_config.gradient_accumulation_steps > 1:
                    loss = loss / self.training_config.gradient_accumulation_steps

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

                # self.tr_loss += loss.item()  # for local model train
                pbar.set_postfix({'Global model train loss': '{0:1.5f}'.format(loss)})
                pbar.update(1)

                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.training_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.training_config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule

                    # self.global_step += 1

                # calculate training results
                if self.model_config.model_output_mode == "token_classification_crf":
                    self.total += label.size(0) * label.size(1)
                    self.correct += (pred_tags == label).sum().item()


        return self.model_parameters  # global model update

        # load local model to train
    def _on_epoch_local(self, train_loader, optimizer, scheduler, global_model_parameters, idx):
        SerializationTool.deserialize_model(self._model, global_model_parameters)
        frz_model_params = [deepcopy(param) for param in self._model.parameters()]

        SerializationTool.deserialize_model(self._model, self.loc_cur_params[idx])
        with tqdm(total=len(train_loader), desc=f"Train rank {self.rank} local model") as pbar:
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

                loss_origin, logits = outputs[:2]
                pred_tags = self.model.backbone.crf.decode(logits, inputs['attention_mask'])

                # ditto loss
                prox_loss = 0
                for w0, w in zip(frz_model_params, self._model.parameters()):
                    prox_loss += torch.sum(torch.pow(w - w0, 2))
                loss = loss_origin + 0.5 * self.federated_config.prox_mu * prox_loss

                optimizer.zero_grad()
                if self.training_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.training_config.gradient_accumulation_steps > 1:
                    loss = loss / self.training_config.gradient_accumulation_steps

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
                pbar.set_postfix({'Local model train loss': '{0:1.5f}'.format(loss)})
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
                if self.model_config.model_output_mode == "token_classification_crf":
                    self.total += label.size(0) * label.size(1)
                    self.correct += (pred_tags == label).sum().item()


        self.loc_cur_params[idx] = self.model_parameters
        return self.model_parameters

    def _on_epoch_end(self, idx):
        """on epoch end"""

        self.logger.info(f"{self.data_config.task_name.upper()} Train, "
                         f"Client:{idx}, Loss:{self.tr_loss / self.global_step:.3f}, "
                         f"Accuracy:{self.correct / self.total:.3f}")

        if not self.federated_config.pson or self.round % self.federated_config.pson_round_len != 0:
            # not need for local test
            return

        valid_data = self._get_dataloader(dataset=self.valid_dataset, client_id=idx)

        # in case not set to local current model
        SerializationTool.deserialize_model(self._model, self.loc_cur_params[idx])
        result = self.eval.test_and_eval(
            model=self._model,  # for ditto, use trained local model
            valid_dl=valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        metric_value = result[self.metric_to_check]
        self.logger.warning(f"metric value for epoch end, {metric_value}")
        operator = np.greater if self.training_config.greater_is_better else np.less
        if (
                self.loc_best_metric.get(idx, None) is None
                or operator(metric_value, self.loc_best_metric[idx])
        ):
            self.loc_best_metric[idx] = metric_value
            self.loc_best_params[idx] = SerializationTool.serialize_model(self._model)
            self.loc_patient_times = 0
            self.logger.warning(f"loc best metric update, {self.loc_best_metric[idx]}")
        else:
            self.loc_patient_times += 1

        self.logger.debug(f"{self.data_config.task_name.upper()} Eval, "
                          f"Client:{idx} epoch end, Local Eval Loss:{result['eval_loss']:.3f}, "
                          f"Current {self.metric_name}:{result[self.metric_name]:.3f}, "
                          f"Best metric {self.metric_to_check}:{self.loc_best_metric[idx]:.3f}")
        # hard-code: to print all metrics for generation task
        if self.model_config.model_output_mode == 'seq_generation':
            log_info = f"Client {idx}, Eval "
            for key, value in result.items():
                log_info += f"metric {key}: {value:.3f};"
            self.logger.debug(log_info)

        if self.loc_patient_times >= self.training_config.patient_times:
            self.stop_early = True

        if self.rank > 0 and self.round % self.federated_config.pson_log_test_len == 0:  # personalized test log, only use for federated
            eval_result_dict = self.test_on_client_locally(idx)  # local test for this client
            metric_title = f"ep_{self.round}" if self.rank < 0 else f"round_{self.round}"
            self.metric_log[metric_title][f"client {idx}"] = eval_result_dict[idx]

    def test_on_client_locally(self, assigned_idx=None):
        if not self.federated_config.pson and len(self.loc_best_params) == 0:
            self.logger.warning("There is no best global params for each client. "
                                "We will take the final local model of each client "
                                "to test for this local algorithm.")
            for i, params in self.loc_cur_params.items():
                self.loc_best_params[i] = params

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


class DittoClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)

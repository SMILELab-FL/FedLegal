"""federated average client"""

from abc import ABC
from copy import deepcopy

import torch
from tqdm import tqdm

from trainers.BaseClient import BaseClientTrainer, BaseClientManager, CRFClientTrainer


class FedAvgClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, test_dataset):
        super().__init__(model, train_dataset, valid_dataset, test_dataset)

class FedAvgClientTrainerCRF(CRFClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, test_dataset):
        super().__init__(model, train_dataset, valid_dataset, test_dataset)

    def _on_epoch(self, train_loader, optimizer, scheduler):
        frz_model_params = deepcopy([param for param in self._model.parameters()])

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
                if self.model_config.model_output_mode == "token_classification_crf":
                    self.total += label.size(0) * label.size(1)
                    self.correct += (pred_tags == label).sum().item()


class FedAvgClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)

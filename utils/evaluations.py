"""define evaluation scripts for FedETuning """

import torch
import numpy as np
from tqdm import tqdm
from abc import ABC
from collections import defaultdict
from transformers import TextGenerationPipeline

from utils.register import registry


class BaseEval(ABC):
    def __init__(self, device, metric):
        self.device = device
        self.metric = metric
        self.task_name = metric.task_name
        self.logger = registry.get("logger")
        config = registry.get("config")
        self.training_config = config.training_config
        self.model_config = config.model_config

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        raise NotImplementedError


@registry.register_eval("glue")
class GlueEval(BaseEval, ABC):
    def __init__(self, device, metric):
        super(GlueEval, self).__init__(device, metric)

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        model.to(self.device)

        eval_loss, nb_eval_steps = 0.0, 0
        preds, out_label_ids = None, None
        results = {}

        for batch in valid_dl:
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]
                          }
                if model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = \
                        batch[2] if model_type in ['bert', 'xlnet'] else None
                outputs = model(inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss
        if model_output_mode == "seq_classification":
            preds = np.argmax(preds, axis=1)
        elif model_output_mode == "seq_regression":
            preds = np.squeeze(preds)

        result = self.metric.calculate_metric(preds, out_label_ids, False)
        results.update(result)

        return results


@registry.register_eval("conll")
class CoNLLEval(BaseEval, ABC):
    def __init__(self, device, metric):
        super(CoNLLEval, self).__init__(device, metric)

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        model.to(self.device)

        eval_loss, nb_eval_steps = 0.0, 0
        preds, out_label_ids = None, None
        results = {}

        for batch in valid_dl:
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]
                          }
                if model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = \
                        batch[2] if model_type in ['bert', 'xlnet'] else None
                outputs = model(inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss
        if model_output_mode == "seq_classification":
            preds = np.argmax(preds, axis=1)
        elif model_output_mode == "seq_regression":
            preds = np.squeeze(preds)
        elif model_output_mode == "token_classification":
            preds = np.argmax(preds, axis=-1)

        label_list = registry.get("id2label")
        result = self.metric.calculate_metric(preds, out_label_ids, label_list, False)
        results.update(result)

        return results


@registry.register_eval("legal")
class LegalEval(BaseEval, ABC):
    def __init__(self, device, metric):
        super(LegalEval, self).__init__(device, metric)

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        model.to(self.device)

        eval_loss, nb_eval_steps = 0.0, 0
        preds, out_labels = None, None
        results = {}
        metric_attr = {
            "crf_process": 'crf' in model_output_mode,
            'multi_label': 'multi' in model_output_mode,
            "multi_label_threshold": self.training_config.multi_label_threshold,
            'analysis': self.training_config.analysis
        }
        tokenizer = registry.get("tokenizer")
        if metric_attr['analysis']:
            record_inputs = []

        with tqdm(total=len(valid_dl), desc=f"Test and Eval") as pbar:
            for step, batch in enumerate(valid_dl):
                # if step >= 2:  # debug
                #     break
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[3]
                              }
                    if model_type != 'distilbert':
                        # XLM, DistilBERT and RoBERTa don't use segment_ids
                        inputs['token_type_ids'] = \
                            batch[2] if model_type in ['bert', 'xlnet'] else None

                    if model_output_mode == 'seq_generation':
                        tmp_eval_loss = torch.tensor([0.0])
                    else:
                        outputs = model(inputs)
                        tmp_eval_loss, logits = outputs[:2]
                        eval_loss += tmp_eval_loss.mean().item()

                nb_eval_steps += 1
                pbar.set_postfix({'loss': '{0:1.5f}'.format(tmp_eval_loss.mean().item())})
                pbar.update(1)

                if metric_attr['multi_label']:
                    pred_item = torch.sigmoid(logits).detach().cpu().numpy()
                elif model_output_mode == 'seq_generation':
                    pred_item, ref_item = [], []
                    for input_item in inputs['input_ids']:
                        seq_index_cond = (input_item == tokenizer.sep_token_id).nonzero()[0]
                        output_seq = model.generate(torch.unsqueeze(input_item[: seq_index_cond], dim=0))
                        generated_seq = output_seq[0][0]
                        pred_text = tokenizer.decode(generated_seq[seq_index_cond:], skip_special_tokens=True)
                        ref_text = tokenizer.decode(input_item[seq_index_cond:], skip_special_tokens=True)

                        pred_item.append(pred_text)
                        ref_item.append(ref_text)
                elif metric_attr['crf_process']:
                    pred_tags = model.backbone.crf.decode(logits, inputs['attention_mask'])[
                        0].detach().cpu().numpy()  # choose the best
                    pred_item = pred_tags
                else:
                    pred_item = logits.detach().cpu().numpy()

                # concat pred_item
                if preds is None:
                    if model_output_mode == 'seq_generation':
                        preds = pred_item
                        out_labels = ref_item
                    else:
                        preds = pred_item
                        out_labels = inputs['labels'].detach().cpu().numpy()
                else:
                    if model_output_mode == 'seq_generation':
                        preds = preds + pred_item  # list concat
                        out_labels = out_labels + ref_item
                    else:
                        preds = np.append(preds, pred_item, axis=0)
                        out_labels = np.append(out_labels, inputs['labels'].detach().cpu().numpy(), axis=0)

                # record input item:
                if metric_attr['analysis']:
                    for input_item in inputs['input_ids']:
                        if 'token_classification' in model_output_mode:
                            temp_text = [word for word in tokenizer.decode(input_item, skip_special_tokens=False).split(" ")
                                         if word != tokenizer.pad_token]
                            record_inputs.append(temp_text)
                        else:
                            record_inputs.append(tokenizer.decode(input_item, skip_special_tokens=True).replace(" ",""))

            eval_loss = eval_loss / nb_eval_steps
            results["eval_loss"] = eval_loss

        label_list = None
        if model_output_mode == "seq_classification":
            preds = np.argmax(preds, axis=1)
        elif model_output_mode == "seq_regression":
            preds = np.squeeze(preds)
        elif "token_classification" in model_output_mode:
            if not metric_attr['crf_process']:
                preds = np.argmax(preds, axis=-1)
            label_list = registry.get("id2label")

        self.metric.set_attribute(**metric_attr)

        if metric_attr['analysis']:
            result = self.metric.calculate_metric(preds, out_labels, label_list, False, loss=eval_loss, input_texts=record_inputs)
        else:
            result = self.metric.calculate_metric(preds, out_labels, label_list, False, loss=eval_loss)
        results.update(result)

        return results

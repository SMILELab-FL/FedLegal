import logging
from collections import defaultdict

import numpy as np
from abc import ABC

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report

from utils.bleu_scorer import BleuScorer, Bleu
from utils.register import registry
from tools.glue_scripts.glue_metric import glue_compute_metrics
from sklearn.metrics import f1_score as f1_score_skl
from sklearn.metrics import r2_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from math import log
from bert_score import BERTScorer


class BaseMetric(ABC):
    def __init__(self, task_name, greater_is_better=False):
        super().__init__()

        self.task_name = task_name
        self.results = {}

    def calculate_metric(self, *args):
        raise NotImplementedError

    def update_metrics(self, *args):
        raise NotImplementedError

    @property
    def best_metric(self):
        return self.results

    @property
    def metric_name(self):
        raise NotImplementedError


@registry.register_metric("glue")
class GlueMetric(BaseMetric):
    def __init__(self, task_name, greater_is_better=False):
        super().__init__(task_name, greater_is_better)

    def calculate_metric(self, preds, labels, updated=True):
        results = glue_compute_metrics(self.task_name, preds, labels)
        if updated:
            self.update_metrics(results)
        return results

    def update_metrics(self, results):

        cur_valid_metric = results[self.metric_name]
        if self.greater_is_better:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric

        if is_best:
            self.results.update(results)
            self.best_valid_metric = cur_valid_metric

    @property
    def metric_name(self):

        glue_metric_name = {
            "cola": "mcc",
            "sst-2": "acc",
            "mrpc": "f1",
            "sts-b": "corr",
            "qqp": "acc",
            "mnli": "acc",
            "mnli-mm": "acc",
            "qnli": "acc",
            "rte": "acc",
            "wnli": "acc"
        }

        return glue_metric_name[self.task_name]


@registry.register_metric("conll")
class CoNLLMetric(BaseMetric):
    def __init__(self, task_name, greater_is_better=False):
        super().__init__(task_name, greater_is_better)

    def calculate_metric(self, preds, labels, label_list, updated=True):

        predictions, labels = preds, labels

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = {
            "accuary": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }

        if updated:
            self.update_metrics(results)

        return results

    def update_metrics(self, results):

        cur_valid_metric = results[self.metric_name]
        if self.greater_is_better:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric

        if is_best:
            self.results.update(results)
            self.best_valid_metric = cur_valid_metric

    @property
    def metric_name(self):
        return "f1"


@registry.register_metric("legal")
class LegalMetric(BaseMetric):
    def __init__(self, task_name, greater_is_better=False):
        super().__init__(task_name, greater_is_better)
        self.fix_metric_greater_relation()  # TODO: hard code to solve different relation when evaluation by different metric

    def fix_metric_greater_relation(self):
        if 'loss' in self.metric_name:
            self.greater_is_better = False
        else:
            self.greater_is_better = True
        self.best_valid_metric = -float("inf") if self.greater_is_better else float("inf")

    def set_attribute(self, **kwargs):
        """For crf, multi_label processing"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def calculate_metric(self, preds, labels, label_list=None, updated=True, **kwargs):
        results = self.legal_compute_metrics(self.task_name, preds, labels, label_list, **kwargs)

        if updated:
            self.update_metrics(results)

        return results

    def update_metrics(self, results):
        cur_valid_metric = results[self.metric_name]
        if self.greater_is_better:
            is_best = cur_valid_metric > self.best_valid_metric
        else:
            is_best = cur_valid_metric < self.best_valid_metric

        if is_best:
            self.results.update(results)
            self.best_valid_metric = cur_valid_metric
        return self.best_valid_metric

    @property
    def metric_name(self):
        legal_metric_name = {
            "lcp": "f1_micro",
            "ljp": "score",
            "ler": "f1_micro",
            "lre": "f1_micro_pos",
            "lam": "f1_micro",
            "ldg": "bleu2"
        }

        return legal_metric_name[self.task_name]

    @property
    def metric_log_skip_name(self):
        legal_metric_skip_name_list = {
            "lcp": ['analysis'],
            "ljp": ['analysis'],
            "ler": ['analysis'],
            "lre": ['analysis', 'analysis_pos'],
            "lam": ['analysis'],
            "ldg": ['hyp', 'ref', 'error_item']
        }

        return legal_metric_skip_name_list[self.task_name]

    def legal_compute_metrics(self, task_name, preds, labels, label_list, **kwargs):
        if preds is not None and labels is not None:
            assert len(preds) == len(
                labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"

        if task_name == "lcp":
            return self.acc_and_f1(preds, labels, **kwargs)
        elif task_name == "lre":
            all_result = self.acc_and_f1(preds, labels)  # for training and eval metric
            if '无' in label_list:  # negative sampling
                mask = labels != 0
                preds = preds[mask]
                labels = labels[mask]
                if 'input_texts' in kwargs:
                    kwargs['input_texts'] = [kwargs['input_texts'][i] for i, value in enumerate(mask) if value]
                pos_result = self.acc_and_f1(preds, labels, **kwargs)
                for key, value in pos_result.items():
                    all_result[key + '_pos'] = value
            return all_result
        elif task_name == "ljp":
            return self.ljp_log_s(preds, labels, **kwargs)
        elif task_name == "ler":
            return self.ner_acc_and_f1(preds, labels, label_list, **kwargs)
        elif task_name == "lam":
            return self.multi_acc_f1(preds, labels, **kwargs)
        elif task_name == "ldg":
            return self.LM_bleu_rouge(preds, labels, **kwargs)
        else:
            raise KeyError(task_name)

    def acc_and_f1(self, preds, labels, **kwargs):
        acc = (preds == labels).astype(np.int32).mean()
        f1_micro = f1_score_skl(y_true=labels, y_pred=preds, average='micro')
        return {
            "acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_score_skl(y_true=labels, y_pred=preds, average='macro'),
            "acc_and_f1": (acc + f1_micro) / 2,
            "analysis": self.analysis_check(0, preds, labels, **kwargs)
        }

    def multi_acc_f1(self, preds, labels, **kwargs):
        preds = (np.array(preds) >= self.multi_label_threshold).astype(np.int32)
        acc = (preds == labels).astype(np.int32).mean()
        # f1_samples = f1_score_skl(y_true=labels, y_pred=preds, average='samples')
        return {
            "acc": acc,
            "f1_micro": f1_score_skl(y_true=labels, y_pred=preds, average='micro'),
            "f1_macro": f1_score_skl(y_true=labels, y_pred=preds, average='macro'),
            # "f1_samples": f1_samples,
            # "acc_and_f1": (acc + f1_samples) / 2,
            "analysis": self.analysis_check(1, preds, labels, **kwargs)
        }

    def ner_acc_and_f1(self, preds, labels, label_list=None, **kwargs):
        pad_tag = 0 if self.crf_process else -100
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != pad_tag]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != pad_tag]
            for prediction, label in zip(preds, labels)
        ]

        return {
            "acc": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1_micro": f1_score(true_labels, true_predictions, average='micro'),
            "f1_macro": f1_score(true_labels, true_predictions, average='macro'),
            "analysis": self.analysis_check(1, true_predictions, true_labels, **kwargs)
        }

    def ljp_log_s(self, preds, labels, **kwargs):
        def get_sc(v1, v2):
            try:
                v = abs(log(v1 + 1) - log(v2 + 1))
            except:
                print("Score computing error!", v1, v2)
                return 0
            if v <= 0.2:
                sc = 1
            elif v <= 0.4:
                sc = 0.8
            elif v <= 0.6:
                sc = 0.6
            elif v <= 0.8:
                sc = 0.4
            elif v <= 1.0:
                sc = 0.2
            else:
                sc = 0
            return sc

        score = [get_sc(p, l) for p, l in zip(preds, labels)]

        acc_01, acc_02 = 0, 0
        for i in range(len(preds)):
            if labels[i] * (1 + 0.1) >= preds[i] >= labels[i] * (1 - 0.1):
                acc_01 += 1
            if labels[i] * (1 + 0.2) >= preds[i] >= labels[i] * (1 - 0.2):
                acc_02 += 1
        acc_01 /= len(labels)
        acc_02 /= len(labels)

        kwargs['compare_metric'] = score
        kwargs['thred'] = np.mean(score)
        return {
            'score': np.mean(score),
            'em': np.sum(preds == labels) / len(labels),
            'acc@0.1': acc_01,
            'acc@0.2': acc_02,
            'r^2': r2_score(labels, preds),
            "analysis": self.analysis_check(2, preds, labels, **kwargs)
        }

    def LM_bleu_rouge(self, preds, labels):
        bleu2_result, bleu3_result = [], []
        rouge_result, bert_score_result = {'p': [], 'r': [], 'f': []}, {'p': [], 'r': [], 'f': []}
        error_item = {'index': [], 'hyp': [], 'ref': []}
        rouge = Rouge()
        smoothie = SmoothingFunction().method4
        bert_scorer = BERTScorer(lang='zh', rescale_with_baseline=True)
        for prediction, reference in zip(preds, labels):
            if len(prediction) == 0:
                bleu2_result.append(0)
                bleu3_result.append(0)
                for key, value in rouge_result.items():
                    rouge_result[key].append(0)
            else:
                try:
                    ref = [reference.split()]
                    hyp = prediction.split()
                    bleu2_result.append(sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
                    bleu3_result.append(
                        sentence_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))

                    rouge_score = rouge.get_scores(prediction, reference)[0]["rouge-l"]
                    for key, value in rouge_score.items():
                        rouge_result[key].append(value)
                except:
                    # in case of error of rouge computation for empty hpothesis
                    error_item['index'].append(len(bleu2_result))
                    error_item['hyp'].append(prediction)
                    error_item['ref'].append(reference)

                    print(f"error index: {len(bleu2_result)}")
                    print(f"hpothesis/pred: {prediction}")
                    print(f"ref: {reference}")
                    bleu2_result.append(0)
                    bleu3_result.append(0)
                    for key, value in rouge_result.items():
                        rouge_result[key].append(0)

        bert_score_result['p'], bert_score_result['r'], bert_score_result['f1'] = bert_scorer.score(preds, labels)

        return {
            'analysis': bleu2_result,
            'bleu2': np.mean(bleu2_result),
            'bleu3': np.mean(bleu3_result),
            'rouge-l': np.mean(rouge_result['f']),
            'bert_score_result': bert_score_result['f1'].mean(),
            'error_item': error_item
        }

    # checking results for case study
    def analysis_check(self, c_type, preds, labels, **kwargs):
        if self.analysis:
            if self.task_name == 'lre' and len(kwargs) == 0:
                return None
            assert 'input_texts' in kwargs.keys()
            label_check = defaultdict(dict)
            input_texts = kwargs['input_texts']

            for i, (pred, label, text) in enumerate(zip(preds, labels, input_texts)):
                if c_type == 0:  # confusion metric
                    item = label_check[f'label_{label}']
                    if f'pred_{pred}' not in item:
                        item[f'pred_{pred}'] = []

                    if pred == label:
                        item[f'pred_{pred}'] = [item[f'pred_{pred}'][0]+1] if len(item[f'pred_{pred}']) > 0 else [1]
                    else:
                        item[f'pred_{pred}'].append({
                            'input_text': text,
                            'pred': pred,
                            'label': label
                        })
                elif c_type == 1:  # multi label
                    for l_idx, (p, l) in enumerate(zip(pred, label)):
                        item = label_check[f'class_{l_idx}']
                        if f'label_{l}_pred_{p}' not in item:
                            item[f'label_{l}_pred_{p}'] = []

                        if p == l:
                            if len(item[f'label_{l}_pred_{p}']) == 0:
                                item[f'label_{l}_pred_{p}'] = [1]
                            else:
                                item[f'label_{l}_pred_{p}'] += 1
                        else:
                            item[f'label_{l}_pred_{p}'].append({
                                'input_text': text,
                                'pred': pred,
                                'label': label
                            })
                else:  # pred value thred
                    assert 'thred' in kwargs.keys() and 'compare_metric' in  kwargs.keys()
                    if kwargs['thred'] not in label_check['over_thred']:
                        label_check['over_thred'][kwargs['thred']] = []

                    if kwargs['compare_metric'][i] < kwargs['thred']:
                        label_check['over_thred'][kwargs['thred']].append({
                            'input_text': text,
                            'pred': pred,
                            'label': label
                        })
            return label_check

        return None

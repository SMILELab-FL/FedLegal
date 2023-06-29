# encoding=utf8
import argparse
import copy
import json
import os
import pickle
from collections import defaultdict

import torch
from loguru import logger
from sklearn.model_selection import train_test_split
import re
import numpy as np
import csv
import random
from queue import Queue
import sys

from tqdm import tqdm

sys.path.append("../../")
from tools.legal_scripts.partition import LegalDataPartition
from utils import pickle_write, make_sure_dirs, setup_seed


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, required=True, help="run directory of user machine")
    parser.add_argument("--data_dir", default='/datasets/legal/data', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", type=str, required=True, help="Task name")
                        # LCP, LJP, LER, LRE, LAM
    parser.add_argument("--output_dir", default='/datasets/legal', type=str,
                        help="The output directory to save partition or raw data")
    parser.add_argument("--clients_num", default=None, type=int,
                        help="All clients numbers")
    parser.add_argument("--alpha", default=None, type=float,
                        help="The label skew degree.")
    parser.add_argument("--overwrite", default=False, type=bool,
                        help="overwrite")
    parser.add_argument("--local_test", default=None, type=int,
                        help="construct local test")
    parser.add_argument("--seed", default=42, type=int,
                        help="seed")
    parser.add_argument("--cluster", action='store_true',
                        help="Use embedding clustering to partition")
    parser.add_argument("--train_neg_num", default=0, type=int,
                        help="Whether to use negative sampling for LRE")

    args = parser.parse_args()
    return args


class LegalProcess:
    def __init__(self, seed=42):
        self.label_list, self.silo_list, self.output_mode = [], [], None
        self.seed = seed

    def _load_raw_data(self, file_path):
        self.dataset = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]

    def _natural_train_test_split(self):
        self.clients_examples, glo_val_test_examples = train_test_split(self.all_examples, test_size=0.2,
                                                                        random_state=self.seed)
        self.global_valid_examples, self.global_test_examples = train_test_split(glo_val_test_examples, test_size=0.5,
                                                                                 random_state=self.seed)
        self.clients_example_dict = {'train': defaultdict(list),
                                     'valid': defaultdict(list),
                                     'test': defaultdict(list),
                                     'all': defaultdict(list)}
        # collect all clients' train as global train data
        self.global_train_examples = []
        for example in self.clients_examples:
            client_idx = self.silo_list.index(example['silo'])
            self.clients_example_dict['all'][client_idx].append(example)

        for client_idx in range(len(self.silo_list)):
            self.clients_example_dict['train'][client_idx], valid_test_temp = \
                train_test_split(self.clients_example_dict['all'][client_idx], test_size=0.2, random_state=self.seed)
            self.clients_example_dict['valid'][client_idx], self.clients_example_dict['test'][client_idx] = \
                train_test_split(valid_test_temp, test_size=0.5, random_state=self.seed)
            self.global_train_examples = self.global_train_examples + self.clients_example_dict['train'][client_idx]

        self.clients_example_dict.pop('all')

    def get_data_info(self):
        return {
            "global_train": self.global_train_examples, "global_valid": self.global_valid_examples,
            "global_test": self.global_test_examples,
            "output_mode": self.output_mode, "label_list": self.label_list, "silo_list": self.silo_list,
            "clients_example_dict": self.clients_example_dict
        }

    def get_labels(self):
        return self.label_list

    def get_examples(self, split):
        if split == 'global_train':
            return self.train_examples
        elif split == 'global_valid':
            return self.global_valid_examples
        elif split == 'global_test':
            return self.global_test_examples
        elif split == 'clients':
            return self.clients_example_dict
        raise Exception("Unknown split :{}".format(split))

    def get_silo_list(self):
        return self.silo_list


class LcpProcessor(LegalProcess):
    def __init__(self, data_dir, data_file, seed=42):
        super(LcpProcessor, self).__init__(seed)
        self._load_raw_data(os.path.join(data_dir, data_file))
        self._process_label()
        self._natural_train_test_split()
        self.output_mode = 'seq_classification'

    def _process_label(self):
        self.all_examples, pre_examples = [], []
        pre_court2count, self.silo2count = defaultdict(int), defaultdict(int)

        for idx, item in enumerate(self.dataset):
            if len(item['claims']) > 512:
                continue
            if item['etl_case_cause'] == '' or item['etl_case_cause'] is None:
                continue

            court = item['ws_court']
            pre_examples.append({
                'id': str(idx),
                'text_a': item['claims'],
                'text_b': "",
                'label': item['etl_case_cause'],
                # 'silo': prov,
                'silo': court
            })
            pre_court2count[court] += 1

        # filter and downsampling
        down_sample_pre_data = defaultdict(list)
        down_sample_label_count = defaultdict(dict)

        for ex in pre_examples:
            if pre_court2count[ex['silo']] < 1000:  # small sum_count filter
                continue
            if pre_court2count[ex['silo']] > 20000:  # down-sampling later
                down_sample_pre_data[ex['silo']].append(ex)
                if ex['label'] not in down_sample_label_count[ex['silo']]:
                    down_sample_label_count[ex['silo']][ex['label']] = 1
                else:
                    down_sample_label_count[ex['silo']][ex['label']] += 1
                continue

            # saving for use
            if ex['label'] not in self.label_list:
                self.label_list.append(ex['label'])
            if ex['silo'] not in self.silo_list:
                self.silo_list.append(ex['silo'])

            self.all_examples.append(ex)
            self.silo2count[ex['silo']] += 1

        # down-sampling
        for silo, pre_sampled_list in down_sample_pre_data.items():
            n_samples = 20000
            sum_down_sample_data = sum(list(down_sample_label_count[silo].values()))
            for label_name, label_count in down_sample_label_count[silo].items():
                down_sample_label_count[silo][label_name] = int((label_count / sum_down_sample_data) * n_samples)

            after_sample_label2count = defaultdict(int)
            random.shuffle(pre_sampled_list)
            for ex in pre_sampled_list:
                if after_sample_label2count[ex['label']] < down_sample_label_count[silo][ex['label']]:
                    if ex['label'] not in self.label_list:
                        self.label_list.append(ex['label'])
                    if ex['silo'] not in self.silo_list:
                        self.silo_list.append(ex['silo'])

                    self.all_examples.append(ex)
                    after_sample_label2count[ex['label']] += 1
                    self.silo2count[ex['silo']] += 1

        print(self.silo2count)


class LjpProcessor(LegalProcess):
    def __init__(self, data_dir, data_file, seed=42):
        super(LjpProcessor, self).__init__(seed)
        self._load_raw_data(os.path.join(data_dir, data_file))
        self.time_dict = {
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
            "十一": 11,
            "十二": 12,
            "十三": 13,
            "十四": 14,
            "十五": 15,
            "十六": 16,
            "十七": 17,
            "十八": 18,
            "十九": 19,
            "二十": 20,
            "二十一": 21,
            "二十二": 22,
            "二十三": 23,
            "二十四": 24,
            "二十五": 25,
            "二十六": 26,
            "二十七": 27,
            "二十八": 28,
            "二十九": 29
        }
        self._process_label()
        self._natural_train_test_split()
        self.output_mode = 'seq_regression'
        self.label_list = [0]  # dummy label for regression

    def _process_label(self):
        pattern = re.compile("经审理查明")
        # prov_pattern = "(.*省|.*自治区|上海|北京|天津|重庆)"
        # prov_pattern = "(河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|北京|天津|上海|重庆|香港|澳门)"
        prov_pattern = "(河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|四川|贵州|云南|甘肃|广西|宁夏|新疆|北京|上海)"  # skip < 1000

        self.all_examples = []
        self.silo2count = defaultdict(int)
        self.silo2examples = defaultdict(list)
        uuid_set = set()
        for idx, item in enumerate(self.dataset):
            if '法院' not in item['segments'].keys():
                continue
            prov_match = re.search(prov_pattern, item['segments']['法院'])
            if prov_match:
                item['prov'] = prov_match.group(1)
            else:
                continue

            parsed = item['parsed']

            if not pattern.search(parsed["事实认定"]):
                continue
            if len(parsed['事实认定']) > 512:  # filter too long text
                continue
            kept_flag = True
            for defedant in parsed["判决结果"]:
                for charge in defedant["构成罪名"]:
                    if "有期徒刑" not in charge["判决结果"] or "缓刑" in charge["判决结果"]:
                        kept_flag = False
                        break

                    extracted = self._extract_prison_term(charge["判决结果"])
                    if sum(extracted) == 0:
                        kept_flag = False
                        break

                    charge["标准刑期"] = extracted
                if not kept_flag:
                    break

            if not kept_flag:
                continue

            if parsed['ws_uuid'] in uuid_set:
                continue
            uuid_set.add(parsed['ws_uuid'])

            charge_result = parsed['判决结果'][0]  # choose the first to control single-defendant
            self.all_examples.append({
                'id': parsed['ws_uuid'],
                'text_a': charge_result['被告姓名'] + ";" + charge_result['构成罪名'][0]['罪名'],
                'text_b': parsed['事实认定'],
                'label': charge_result['构成罪名'][0]['标准刑期'][-1],
                # 'court': item['segments']['法院'],
                'silo': item['prov']
            })
            if item['prov'] not in self.silo_list:
                self.silo_list.append(item['prov'])
            self.silo2count[item['prov']] += 1
            self.silo2examples[item['prov']].append(idx)

    def _extract_prison_term(self, string):
        year_term = 0
        pattern_year = "([一二三四五六七八九十]{1,3})年"
        year_match = re.search(pattern_year, string)
        if year_match:
            try:
                year_term = self.time_dict[year_match.group(1)]
            except KeyError:
                return 0, 0, 0

        month_term = 0
        pattern_month = "([一二三四五六七八九十]{1,3})个月"
        month_match = re.search(pattern_month, string)
        if month_match:
            try:
                month_term = self.time_dict[month_match.group(1)]
            except KeyError:
                return 0, 0, 0
        return year_term, month_term, year_term * 12 + month_term


class LerProcessor(LegalProcess):
    def __init__(self, data_dir, data_file, seed=42):
        super(LerProcessor, self).__init__(seed)
        self._load_raw_data(os.path.join(data_dir, data_file))
        self._process_label()
        self._natural_train_test_split()
        self.output_mode = 'token_classification'

    def _process_label(self):
        self.bio_suffix = set('O')
        self.label_list = ['X', 'O']  # X is used for special token with mask=0 in tokenizer and label alignment
        self.all_examples = []

        self.silo2count = defaultdict(int)
        uuid_set = set()

        # prov_pattern = re.compile("[-|省\-|市\-]")
        for idx, item in enumerate(self.dataset):
            source_data = json.loads(item['source_data'])
            target_data = json.loads(item['target_data'])

            if source_data['uuid'] in uuid_set or target_data is None:
                continue

            # prov_match = re.search(prov_pattern, source_data['uuid'])
            # if prov_match:
            #     print(prov_match)
            #     item['prov'] = prov_match.group(1)

            if source_data['uuid'].find('-') != -1:
                item['prov'] = source_data['uuid'][: source_data['uuid'].find('-')][:2]

            start_i = 0
            while start_i < len(source_data['srcText']):
                end_i = start_i + 512
                text = source_data['srcText'][start_i: end_i]

                item_tokens = list(text)
                item_ner_label = ['O' for _ in range(len(item_tokens))]

                record_next_start = False
                for record in target_data['exact']:  # list
                    if record_next_start:  # break this example in advance
                        break
                    key = record['labelValues'][0]
                    offset = record['offset']

                    if not check_offset_in(offset, start_i, end_i):
                        if offset[0] >= start_i and offset[1] >= end_i:  # if part of entity exceeds
                            item_tokens = item_tokens[: offset[0]]
                            item_ner_label = item_ner_label[: offset[0]]
                            start_i = offset[0]  # as the next example start index
                            record_next_start = True
                        continue

                    if key not in self.bio_suffix:
                        self.bio_suffix.add(key)
                        self.label_list.append('B-' + key)
                        self.label_list.append('I-' + key)

                    item_ner_label[offset[0] - start_i] = 'B-' + key
                    for i in range(offset[0] + 1, offset[1]):
                        item_ner_label[i - start_i] = 'I-' + key

                self.all_examples.append({
                    'id': str(idx),
                    'tokens': item_tokens,
                    'label': item_ner_label,
                    'silo': item['prov']
                })

                if item['prov'] not in self.silo_list:
                    self.silo_list.append(item['prov'])
                self.silo2count[item['prov']] += 1

                if not record_next_start:
                    start_i += 512
            uuid_set.add(source_data['uuid'])

        print(len(self.all_examples))
        print(self.silo2count)


class LreProcessor(LegalProcess):
    def __init__(self, data_dir, data_file, seed=42, train_neg_num=0):
        super(LreProcessor, self).__init__(seed)
        self.train_neg_num = train_neg_num
        self.neg_sample = self.train_neg_num > 0
        self._load_raw_data(os.path.join(data_dir, data_file))
        self._process_label()
        self._natural_train_test_split()
        if self.neg_sample:
            logger.critical(f"Generate {self.train_neg_num} negative examples per training sample and all negative"
                            f" examples per validation/test sample for LRE.")
            self._negative_sampling(self.train_neg_num)
        self.output_mode = 'seq_classification'

    def _process_label(self):
        self.label_list = ['无'] if self.neg_sample else []
        self.all_examples = []

        self.silo2count = defaultdict(int)
        uuid_set = set()

        for idx, item in enumerate(self.dataset):
            source_data = json.loads(item['source_data'])
            target_data = json.loads(item['target_data'])

            if source_data['uuid'] in uuid_set or target_data is None:
                continue

            if source_data['uuid'].find('-') != -1:
                item['prov'] = source_data['uuid'][: source_data['uuid'].find('-')][:2]

            # text = source_data['srcText']
            # iteration every 512
            for i in range(0, len(source_data['srcText']), 512):
                start_i, end_i = i, i + 512
                text = source_data['srcText'][start_i: end_i]
                all_entities_group = []
                all_entities_value = [item['span'][0] for item in target_data['exact']
                                      if check_offset_in(item['offset'], start_i, end_i)]  # filter length limit
                for i in range(len(all_entities_value) - 1):
                    for j in range(i + 1, len(all_entities_value)):
                        if all_entities_value[i] == all_entities_value[j]:  # skip repeated entity
                            continue
                        all_entities_group.append(f'{all_entities_value[i]}与{all_entities_value[j]}')

                # iteration all items
                positive_entities_group = []
                positive_examples = []

                for record in target_data['binaryRelations']:  # list
                    entity_0_id = record['value'].split(',')[0]
                    entity_1_id = record['value'].split(',')[1]
                    entity_0_value, entity_1_value = None, None

                    # search entity span value
                    re_not_in_range = False
                    for ent_item in target_data['exact']:
                        value = ent_item['span'][0]
                        if ent_item['id'] == entity_0_id:
                            if not check_offset_in(ent_item['offset'], start_i, end_i):
                                re_not_in_range = True
                                break
                            entity_0_value = value
                        elif ent_item['id'] == entity_1_id:
                            if not check_offset_in(ent_item['offset'], start_i, end_i):
                                re_not_in_range = True
                                break
                            entity_1_value = value

                    if re_not_in_range:
                        continue

                    positive_examples.append({
                        'id': record['id'],
                        'text_a': entity_0_value + "与" + entity_1_value,
                        'text_b': text,
                        'label': record['type'],
                        'silo': item['prov'],
                    })
                    positive_entities_group.append(entity_0_value + "与" + entity_1_value)
                    positive_entities_group.append(entity_1_value + "与" + entity_0_value)

                    if record['type'] not in self.label_list:
                        self.label_list.append(record['type'])

                    if item['prov'] not in self.silo_list:
                        self.silo_list.append(item['prov'])

                    self.silo2count[item['prov']] += 1

                uuid_set.add(source_data['uuid'])
                # add negative info
                if self.neg_sample:
                    redundant_entities_group = list(set(all_entities_group) - set(positive_entities_group))
                    random.shuffle(redundant_entities_group)  # redundant_entities_group.sort()
                    for example in positive_examples:
                        example['redundant_ent'] = redundant_entities_group

                # add these positive examples into all_examples
                self.all_examples.extend(positive_examples)

        print(self.silo2count)
        print(self.all_examples[:5])

    def _negative_sampling(self, train_neg_num=-1):
        # client train/val/test negative sampling
        for d_type, example_dict in self.clients_example_dict.items():
            logger.info(f"{d_type} dataset negative sampling")
            for client_idx, clients_examples in example_dict.items():
                example_dict[client_idx].extend(negative_sample_for_lists(clients_examples, d_type, train_neg_num))

        # centralized data
        logger.info(f"global dataset negative sampling")
        self.global_train_examples.extend(negative_sample_for_lists(self.global_train_examples, 'train'))
        self.global_valid_examples.extend(negative_sample_for_lists(self.global_valid_examples, 'valid'))
        self.global_test_examples.extend(negative_sample_for_lists(self.global_test_examples, 'test'))


def negative_sample_for_lists(clients_examples, d_type, train_neg_num=-1):
    neg_clients_examples = []

    neg_index = train_neg_num if d_type == 'train' else -1
    for truth_sample in tqdm(clients_examples):
        for neg_text_a in truth_sample['redundant_ent'][:neg_index]:
            neg_sample = copy.deepcopy(truth_sample)
            neg_sample['text_a'] = neg_text_a
            neg_sample['label'] = '无'
            neg_sample.pop('redundant_ent')
            neg_clients_examples.append(neg_sample)
        # truth_sample.pop('redundant_ent')
    return neg_clients_examples


def check_offset_in(offset, start_i, end_i):
    return offset[0] >= start_i and offset[1] < end_i


class LamProcessor(LegalProcess):
    def __init__(self, data_dir, data_file, seed=42):
        super(LamProcessor, self).__init__(seed)
        self._load_raw_data(os.path.join(data_dir, data_file))
        self._process_label()
        self._natural_train_test_split()
        self.output_mode = 'multi_seq_classification'

    def _process_label(self):
        self.all_examples = []
        silo_label_dict = defaultdict(dict)
        for idx, item in enumerate(self.dataset):
            if item['target_data'] is None or len(item['source_data']['srcText']) > 512:
                continue

            if item['cause'] not in self.silo_list:
                self.silo_list.append(item['cause'])

            sent_pairs = item['source_data']['srcText'].split('\n')
            labels = []
            for example in item['target_data']['exact']:
                labels.append(example['labelValues'][0])
                if example['labelValues'][0] not in self.label_list:
                    self.label_list.append(example['labelValues'][0])

            self.all_examples.append({
                'id': str(idx),
                'text_a': sent_pairs[0],
                'text_b': sent_pairs[1],
                'label': labels,
                'silo': item['cause']
            })

            for label in labels:
                if label not in silo_label_dict[item['cause']]:
                    silo_label_dict[item['cause']][label] = 1
                else:
                    silo_label_dict[item['cause']][label] += 1


# Developing Task
class LdgProcessor(LegalProcess):
    def __init__(self, data_dir, data_file, seed=42):
        super(LdgProcessor, self).__init__(seed)
        self._load_raw_data(os.path.join(data_dir, data_file))
        self._natural_train_test_split()
        self.output_mode = 'seq_generation'

    def _load_raw_data(self, file_path):
        self.all_examples, self.silo_list, self.silo2count, self.record_case_list = [], [], defaultdict(int), []
        self.pre_silo_list, self.pre_silo2count = [], defaultdict(int)
        null_fix = False

        with open(file_path, 'r') as csv_file:
            if '\0' in csv_file.read():
                print("You have null bytes in your input file, and we will fix it.")
                null_fix = True

        with open(file_path, 'r') as csv_file:
            if null_fix:
                reader = csv.reader((x.replace('\0', '') for x in csv_file), delimiter='\t')
            else:
                reader = csv.reader(csv_file, delimiter='\t')

            html_filter = re.compile(r'<[^>]+>', re.S)
            for line_idx, row in enumerate(reader):
                if line_idx % 1000 == 0:
                    print(line_idx)

                if line_idx == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_idx += 1
                else:
                    if len(row) > 2:  # and query.search(row[2]):
                        cause = row[0]
                        record_case = row[1]
                        content = html_filter.sub('', row[2]).replace(' ', '').replace("\u3000", '')
                        if '\n' in content:
                            court = re.sub(r'[^\u4e00-\u9fa5]', '', content[:content.find('\n')])
                            if '法院' not in court:
                                continue
                            else:
                                court = court[:court.find('法院') + 2]
                        else:
                            continue  # drop case without place

                        next_line_index = [substr.start() for substr in re.finditer('\n', content)]
                        try:
                            content = content[next_line_index[5]:]  # delete case info
                        except:
                            continue  # if no \n format
                        next_line_index = [substr.start() for substr in re.finditer('\n', content)]  # re compute

                        lawyer_query = '\n[审|长]：'
                        lawyer_text_start_index = [substr.end() for substr in re.finditer(lawyer_query, content)]
                        start_index = 0
                        ex_first_lawyer_index = 0

                        # find the first example to start
                        while ex_first_lawyer_index < len(lawyer_text_start_index) and len(
                                content[start_index: lawyer_text_start_index[ex_first_lawyer_index]]) < 1000:
                            ex_first_lawyer_index += 1

                        for i, content_end_idx in enumerate(lawyer_text_start_index[ex_first_lawyer_index - 1:]):
                            lawyer_ans_end = -1
                            for j in next_line_index:
                                if j > content_end_idx:
                                    lawyer_ans_end = j
                                    break

                            # code for use the complete sentence to start this dialog
                            content_start = lawyer_ans_end - 1024  # max length 1024
                            if content_start > 0:
                                for j in next_line_index:  # first complete sentence start of content，later one of the qualified \n
                                    if j > content_start:
                                        content_start = j + 1
                                        break
                            else:
                                content_start = 0

                            text_a = content[content_start: content_end_idx].replace('\n', '')
                            label = content[content_end_idx: lawyer_ans_end].replace('\n', '')
                            if len(text_a) + len(label) > 1024:
                                continue

                            self.all_examples.append({
                                'text_a': text_a,  # clip the closest 1000 as input
                                'text_b': label,
                                'record_case': record_case,
                                'cause': cause,
                                'silo': court
                            })
                            if record_case not in self.record_case_list:
                                self.record_case_list.append(record_case)
                            if court not in self.pre_silo_list:
                                self.pre_silo_list.append(court)

                            if len(self.all_examples) < 5:
                                print(self.all_examples[-1])

                        if len(self.record_case_list) > 20000:
                            break
            print("pre silo list length", len(self.pre_silo_list))
            print("record case count:", len(self.record_case_list))
            print(f'Processed {line_idx} lines, get {len(self.all_examples)} pre examples')

    def _natural_train_test_split(self):
        random.seed(args.seed)
        random.shuffle(self.record_case_list)
        choose_records = self.record_case_list[:5000]
        clients_records, glo_eval_test_records = train_test_split(choose_records, test_size=0.2, random_state=args.seed)
        glo_valid_records, glo_test_records = train_test_split(glo_eval_test_records, test_size=0.5,
                                                               random_state=args.seed)
        print(clients_records[:5])
        self.global_train_examples, self.global_valid_examples, self.global_test_examples = [], [], []
        self.clients_example_dict = {'train': defaultdict(list),
                                     'valid': defaultdict(list),
                                     'test': defaultdict(list),
                                     'all': defaultdict(list)
                                     }

        # get each records' data samples
        for idx, example in enumerate(self.all_examples):
            rec_case = example['record_case']
            if idx % 100000 == 0:
                print(idx)
            if rec_case in glo_valid_records:
                self.global_valid_examples.append(example)
            elif rec_case in glo_test_records:
                self.global_test_examples.append(example)
            elif rec_case in clients_records:
                client_idx = self.pre_silo_list.index(example['silo'])
                self.clients_example_dict['all'][client_idx].append(example)

        # each client split data
        for i, all_client_exs in enumerate(self.clients_example_dict['all'].values()):
            if i % 1000 == 0:
                print(i)
            if len(all_client_exs) > 2425:
                client_idx = len(self.silo_list)
                self.clients_example_dict['train'][client_idx], temp_valid_test = train_test_split(
                    all_client_exs, test_size=0.2, random_state=args.seed)
                self.clients_example_dict['valid'][client_idx], self.clients_example_dict['test'][
                    client_idx] = train_test_split(
                    temp_valid_test, test_size=0.5, random_state=args.seed)
                self.global_train_examples = self.global_train_examples + self.clients_example_dict['train'][client_idx]
                print(f"{client_idx} train, {len(self.clients_example_dict['train'][client_idx])};"
                      f"valid, {len(self.clients_example_dict['valid'][client_idx])};"
                      f"test, {len(self.clients_example_dict['test'][client_idx])}")
                self.silo_list.append(self.clients_example_dict['train'][client_idx][0]['silo'])
                self.silo2count[client_idx] = len(all_client_exs)

        self.clients_example_dict.pop('all')
        print(f"global train: {len(self.global_train_examples)}; global valid :{len(self.global_valid_examples)}, "
              f"global test: {len(self.global_test_examples)}")


# partition
def load_legal_examples(args):
    if args.task == 'LCP':
        data_file = 'fed_civil.jsonl'
        processor = LcpProcessor(args.data_dir, data_file)
    if args.task == 'LJP':
        data_file = 'fed_criminal.jsonl'
        processor = LjpProcessor(args.data_dir, data_file)
    elif args.task == 'LER':
        # data_file = 'fed_ie_final.jsonl'
        data_file = 'ner-20230307.jsonl'
        processor = LerProcessor(args.data_dir, data_file)
    elif args.task == 'LRE':
        data_file = 'ner-20230307.jsonl'
        processor = LreProcessor(args.data_dir, data_file, train_neg_num=args.train_neg_num)
    elif args.task == 'LAM':
        data_file = 'release_focus.jsonl'
        processor = LamProcessor(args.data_dir, data_file)
    elif args.task == 'LDG':
        data_file = '华宇笔录数据.csv'
        processor = LdgProcessor(args.data_dir, data_file)

    # pickle_write(all_examples, f"{args.output_data_file}")
    print(f"load {len(processor.all_examples)} for {args.task}")

    return processor


def convert_global_to_pkl(args):
    logger.info("reading examples ...")
    if os.path.isfile(args.output_data_file) and not args.overwrite:
        logger.info(f"Examples in {args.output_data_file} have existed ...")
        with open(args.output_data_file, "rb") as file:
            data = pickle.load(file)
    else:
        logger.info(f"Generating examples from {args.data_dir} ...")
        processor = load_legal_examples(args)
        data = processor.get_data_info()
        with open(args.output_data_file, "wb") as file:
            pickle.dump(data, file)
    return data


def convert_legal_to_natural_silo_pkl(args, data, silo_partition="silo"):
    logger.info("partition data ...")
    if os.path.isfile(args.output_partition_file):
        logger.info("loading partition data ...")
        with open(args.output_partition_file, "rb") as file:
            partition_data = pickle.load(file)
    else:
        partition_data = {}

    logger.info(f"partition data's keys: {partition_data.keys()}")

    if silo_partition in partition_data.keys() and not args.overwrite:
        logger.info(f"Partition method 'silo_partition' has existed "
                    f"and overwrite={args.overwrite}, then skip")
    else:
        label_list, silo_list, output_mode, silo_partition_data = data['label_list'], data['silo_list'], \
                                                                  data['output_mode'], data['clients_example_dict']

        label_mapping = {label: idx for idx, label in enumerate(label_list)}
        attribute = {"label_mapping": label_mapping, "label_list": label_list,
                     "clients_num": len(silo_list),
                     "output_mode": output_mode
                     }

        silo_partition_data['attribute'] = attribute
        logger.info(f"writing silo... clients_num={len(silo_list)}")
        partition_data[silo_partition] = silo_partition_data

        with open(args.output_partition_file, "wb") as file:
            pickle.dump(partition_data, file)


def covert_legal_to_heuristic_pkl(args, data):
    logger.info("partition data ...")
    if os.path.isfile(args.output_partition_file):
        logger.info("loading partition data ...")
        with open(args.output_partition_file, "rb") as file:
            partition_data = pickle.load(file)
    else:
        partition_data = {}

    logger.info(f"partition data's keys: {partition_data.keys()}")

    if f"clients={args.clients_num}_alpha={args.alpha}" in partition_data and not args.overwrite:
        logger.info(f"Partition method 'clients={args.clients_num}_alpha={args.alpha}' has existed "
                    f"and overwrite={args.overwrite}, then skip")
    else:
        label_list, silo_list, output_mode, silo_partition_data = data['label_list'], data['silo_list'], \
                                                                  data['output_mode'], data['clients_example_dict']

        all_client_local_train = []
        for train_samples in silo_partition_data['train'].values():
            if args.task == 'LRE':  # skip neg samples:
                pos_train_samples = [ex for ex in train_samples if ex['label'] != '无']
                all_client_local_train = all_client_local_train + pos_train_samples
            else:
                all_client_local_train = all_client_local_train + train_samples

        label_mapping = {label: idx for idx, label in enumerate(label_list)}
        attribute = {"label_mapping": label_mapping, "label_list": label_list,
                     "clients_num": args.clients_num, "alpha": args.alpha,
                     "output_mode": output_mode
                     }

        if args.task == 'LCP' or args.task == 'LRE' or (args.task == 'LER' and not args.cluster):
            partition_method = 'label-skew'
        elif args.cluster:
            partition_method = 'encode-cluster'
        else:
            partition_method = 'quantity-skew'
        logger.info(f"using {partition_method} partition")

        partition_train_data, clients_partition_data_obj = get_partition_data(
            examples=all_client_local_train, num_classes=len(label_list), num_clients=args.clients_num,
            label_vocab=label_list, dir_alpha=args.alpha, partition=partition_method, task=args.task,
            run_dir=args.run_dir
        )

        if clients_partition_data_obj is not None:
            if 'client_cluster_label' not in attribute.keys():
                attribute['client_cluster_label'] = clients_partition_data_obj.cor_cluster_labels

            # add cluster label for silo data
            if 'client_cluster_label' not in silo_partition_data['attribute'].keys():
                count_list = [len(exs) for _, exs in silo_partition_data['train'].items()]
                silo_attr = silo_partition_data['attribute']
                silo_attr['client_cluster_label'] = clients_partition_data_obj.get_cluster_labels(count_list=count_list)
                partition_data['silo']['attribute'] = silo_attr

        clients_partition_data = {"train": partition_train_data,
                                  "valid": silo_partition_data['valid'],
                                  "test": silo_partition_data['test'],
                                  "attribute": attribute
                                  }

        # LRE, neg sampling
        if args.task == 'LRE' and args.train_neg_num > 0:
            logger.info(f"training dataset negative sampling")
            for client_idx, clients_examples in clients_partition_data['train'].items():
                clients_partition_data['train'][client_idx].extend(negative_sample_for_lists(clients_examples, 'train',
                                                                                             train_neg_num=args.train_neg_num))

        logger.info(f"writing clients={args.clients_num}_alpha={args.alpha} ...")
        partition_data[f"clients={args.clients_num}_alpha={args.alpha}"] = clients_partition_data

        # write file
        with open(args.output_partition_file, "wb") as file:
            pickle.dump(partition_data, file)


def get_partition_data(examples, num_classes, num_clients, label_vocab, dir_alpha, partition, task, run_dir):
    if task == 'LER':
        if partition == 'encode-cluster':
            targets = ["".join(example['tokens']) for example in examples]
        else:
            targets = []
            label_vocab = []
            for ex in examples:
                dir_label = []
                for ner_label in ex['label']:
                    if ner_label not in dir_label:
                        dir_label.append(ner_label)
                value = '_'.join([i for i in sorted(dir_label)])
                targets.append(value)
                if value not in label_vocab:
                    label_vocab.append(value)

        num_classes = len(label_vocab)
    else:
        if partition == 'encode-cluster':
            targets = [example['text_b'] for example in examples] if task == 'LRE' \
                else [example['text_a'] for example in examples]
        else:
            targets = [example['label'] for example in examples] if 'label' in examples[0].keys() \
                else [i for i in range(len(examples))]

    clients_partition_data = LegalDataPartition(
        targets=targets, num_classes=num_classes, num_clients=num_clients,
        label_vocab=label_vocab, dir_alpha=dir_alpha, partition=partition, verbose=False, run_prefix=run_dir,
    )
    # assert (len(clients_partition_data) == num_clients,
    #         "The partition function is wrong, please check")
    partition_data = {}
    for client_idx in range(len(clients_partition_data)):
        partition_data[client_idx] = [examples[idx] for idx in clients_partition_data[client_idx]]

    # record the cluster id
    if partition == 'encode-cluster':
        return partition_data, clients_partition_data
    return partition_data, None


if __name__ == "__main__":
    logger.info("start...")
    args = parser_args()
    setup_seed(args.seed)

    run_dir = args.run_dir
    args.data_dir = run_dir + args.data_dir
    args.output_dir = run_dir + args.output_dir

    # silo partition and heuristic partition
    # args.overwrite = False
    output_dir = os.path.join(args.output_dir, "silo")
    make_sure_dirs(output_dir)
    args.output_data_file = os.path.join(output_dir, f"{args.task.lower()}_data.pkl")
    args.output_partition_file = os.path.join(output_dir, f"{args.task.lower()}_partition.pkl")
    logger.info(f"data_dir: {args.data_dir}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"output_data_file: {args.output_data_file}")
    logger.info(f"output_partition_file: {args.output_partition_file}")

    logger.info("global data saving")
    data = convert_global_to_pkl(args)

    logger.info("partition method: natural split, silo")
    convert_legal_to_natural_silo_pkl(args, data, silo_partition="silo")

    client_nums = [10]
    alphas = [0.1, 1.0, 10.0] if args.alpha is None else [args.alpha]
    for client_num in client_nums:
        for alpha in alphas:
            args.alpha = alpha
            args.clients_num = client_num
            logger.info(f"partition method: clients={args.clients_num}_alpha={args.alpha}")
            covert_legal_to_heuristic_pkl(args, data)

    logger.info(f"writing task={args.task} into partition file ...")
    logger.info("end")

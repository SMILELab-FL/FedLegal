import os
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from fedlab.utils.dataset import DataPartitioner, BasicPartitioner
import fedlab.utils.dataset.functional as F
from tools.partitions import label_skew_process, iid_process
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from fedlab.utils.functional import get_best_gpu


class LegalDataPartition(BasicPartitioner):
    def __init__(self, targets, num_clients, num_classes,
                 label_vocab, balance=True, partition="iid",
                 unbalance_sgm=0, num_shards=None,
                 dir_alpha=None, verbose=True, seed=42, run_prefix=None):
        self.targets = targets if partition == 'encode-cluster' else np.array(targets)  # with shape (num_samples,)
        self.num_samples = len(self.targets) if partition == 'encode-cluster' else self.targets.shape[0]
        self.num_clients = num_clients
        self.label_vocab = label_vocab
        self.client_dict = dict()
        self.partition = partition
        self.balance = balance
        self.dir_alpha = dir_alpha
        self.num_shards = num_shards
        self.unbalance_sgm = unbalance_sgm
        self.verbose = verbose
        self.num_classes = num_classes
        self.run_prefix = run_prefix
        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)
        self.seed = seed

        # perform partition according to setting
        self.client_dict = self._perform_partition()
        # get sample number count for each client
        # self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.partition == 'quantity-skew':
            # quantity-skew (Dirichlet)
            client_sample_nums = F.dirichlet_unbalance_split(self.num_clients, self.num_samples,
                                                             self.dir_alpha)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)
        elif self.partition == 'label-skew':
            # label-distribution-skew:distributed-based (Dirichlet)
            # 狄利克雷报错原因：对于总类别list，传入的样本中某些类别不存在，导致占比为0无法进行狄利克雷划分
            client_dict = label_skew_process(
                label_vocab=self.label_vocab, label_assignment=self.targets,
                client_num=self.num_clients, alpha=self.dir_alpha,
                data_length=len(self.targets)
            )
        elif self.partition == 'encode-cluster':
            cluster_num = 5 * self.num_clients  # TODO: hard code
            cluster_labels = self.get_cluster_labels(cluster_num)
            client_dict = label_skew_process(
                label_vocab=list(range(cluster_num)), label_assignment=cluster_labels,
                client_num=self.num_clients, alpha=self.dir_alpha,
                data_length=len(self.targets)
            )

            # Create a dictionary to hold the indices for each label
            self.cor_cluster_labels = {}
            for key, value in client_dict.items():
                self.cor_cluster_labels[key] = [cluster_labels[i] for i in value]
        elif self.partition == 'iid':
            # !!! when datasize is small, it doesn't work
            client_dict = iid_process(data_assignment=self.targets,
                                      client_num=self.num_clients)
        else:
            return None

        # convert np.array to list
        for key, value in client_dict.items():
            if isinstance(value, np.ndarray):
                client_dict[key] = value.tolist()
        return client_dict

    def __getitem__(self, index):
        """Obtain sample indices for client ``index``.

        Args:
            index (int): BaseClient ID.

        Returns:
            list: List of sample indices for client ID ``index``.

        """
        return self.client_dict[index]

    def __len__(self):
        """Usually equals to number of clients."""
        return len(self.client_dict)

    # Perform K-Means clustering
    def _perform_clustering(self, examples, num_clusters, tokenizer, model, device, batch_size=16):
        def collate_fn(examples):
            return tokenizer(examples, padding=True, truncation=True, return_tensors='pt', max_length=512)

        dataloader = DataLoader(examples, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        embeddings = []
        for batch in tqdm(dataloader):
            batch_inputs = batch.to(device)
            batch_outputs = model(**batch_inputs).last_hidden_state[:, 0, :]
            embeddings.append(batch_outputs.detach().cpu().numpy())
        embeddings = torch.from_numpy(np.concatenate(embeddings))
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
        return kmeans.labels_

    def get_cluster_labels(self, cluster_num=None, count_list=None):
        if cluster_num is None:
            cluster_num = 5 * self.num_clients
        model_name = self.run_prefix + "/pretrain/nlp/roberta-wwm-ext"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = get_best_gpu()
        model = AutoModel.from_pretrained(model_name).cuda(device)

        # Example usage
        # examples = ["This is the first example", "This is the second example", "This is the third example"]
        cluster_labels = self._perform_clustering(self.targets, cluster_num, tokenizer, model, device)

        if count_list is not None:
            count_cluster_label_dict = {}
            start_idx = 0
            for cid, count in enumerate(count_list):
                count_cluster_label_dict[cid] = cluster_labels[start_idx: start_idx+count]
                start_idx += count
            return count_cluster_label_dict
        else:
            return cluster_labels  # for label skew


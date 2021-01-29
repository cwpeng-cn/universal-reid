from __future__ import absolute_import
from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class BalancedSampler(Sampler):
    def __init__(self, dataset, P, K):
        super(BalancedSampler, self).__init__((BalancedSampler, self).__init__(dataset))
        self.data = dataset
        self.P = P
        self.K = K
        self.person_indexs_dict = dataset.person_indexs_dict
        self.persons = list(self.person_indexs_dict.keys())
        self.iter_num = len(self.persons) // P

    def __iter__(self):
        random.shuffle(self.persons)
        curr_p = 0
        for it in range(self.iter_num):
            pids = self.persons[curr_p:curr_p + self.P]
            curr_p += self.P
            indexs = []
            for pid in pids:
                if len(self.person_indexs_dict[pid]) >= self.K:
                    sample_index = np.random.choice(self.person_indexs_dict[pid], self.K, False)
                    indexs.extend(sample_index)
                else:
                    sample_index = np.random.choice(self.person_indexs_dict[pid], self.K, True)
                    indexs.extend(sample_index)
            yield indexs

    def __len__(self):
        return self.iter_num

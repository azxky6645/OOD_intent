import os
import re
import html

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import torch

import time


class CustomDataset(Dataset):
    def __init__(self, data_path, known_label_list, tokenizer, max_seq_len, mode):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.unseen_token = 'oos'
        self.label = read_file(os.path.join(data_path, mode, 'label'))
        self.num_labels = len(known_label_list)
        self.unseen_token_id = self.num_labels
        self.known_label_list = known_label_list
        self.label_list = self.known_label_list + [self.unseen_token]
        self.label_dict = {item: idx for idx, item in enumerate(self.label_list)}
        self.label = [i if i in self.known_label_list else self.unseen_token for i in self.label]
        self.label = np.array([self.label_dict[item] for item in self.label])
        self.text = np.array(read_file(os.path.join(data_path, mode, 'seq.in')))
        if mode == 'train':
            drop_ood_idx = np.where(np.array(self.label) == self.label_dict['oos'])[0]
            self.label = np.delete(self.label, drop_ood_idx)
            self.text = np.delete(self.text, drop_ood_idx)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.normalize_string(self.text[idx])
        token = self.tokenizer(text,
                               padding='max_length',
                               max_length=self.max_seq_len,
                               truncation=True
                               )
        token = token.convert_to_tensors(prepend_batch_axis=False, tensor_type='pt')
        label = self.label[idx]
        return token, label

    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s


# def read_file(input_file):
#     with open(input_file, "r", encoding="utf-8") as f:
#         lines = [line.strip() for line in f]
#         return lines


def read_file(input_file):
    df = pd.read_csv(input_file, header=None)
    df = pd.Series(df[0])
    return df


# Iterable dataset의 경우 직접 worker 별로 일을 재분배 해야함
def worker_init_fn():
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]

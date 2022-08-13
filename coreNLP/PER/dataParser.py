'''
Author: Liuhezi
Date: 2021-04-21 15:03:09
LastEditTime: 2021-05-11 11:29:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/PER/dataParser.py
'''
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PERDataset(Dataset):
    def __init__(self, input, device=None):
        self.user_word_adj = np.stack([i for i in input.values()])
        # self.user_word_adj = input
        self.device = device
        self.N = self.user_word_adj.shape[0]

        if self.device is None:
            self.user_word_adj = torch.tensor(self.user_word_adj)
        else:
            self.user_word_adj = torch.tensor(self.user_word_adj).cuda(self.device)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.user_word_adj[idx]

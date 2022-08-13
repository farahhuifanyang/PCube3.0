'''
Author: your name
Date: 2021-04-06 17:57:06
LastEditTime: 2021-07-23 16:49:05
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/NER/data_parser.py
'''


from coreNLP.NER.config import FLAGS
import torch
import torch.utils.data as data


class NERDataset(data.Dataset):
    def __init__(self, inputs, tokenizer, max_length, device=None):
        self.device = device
        self.max_length = max_length - 2
        self.tokenizer = tokenizer

        self.sent_map = {}
        self.datas = []

        # 解析预处理的输入
        for k, v in inputs.items():
            self.sent_map[k] = []
            for s in v:
                self.sent_map[k].append("".join(s["sentence"]))
                self.datas.append(s)

        self.num_example = len(self.datas)

    def loadSent(self, sent):
        # 文件中读取基础数据
        token = list(sent["sentence"])
        if "pos" in FLAGS.features:
            wpos = sent["pos"]
        else:
            wpos = [0]*len(token)

        pos = []
        for t, p in zip(token, wpos):
            pos += [p] * len(t)
        token = list("".join(token))

        # padding
        token_length = len(token)
        pad_length = self.max_length - token_length
        if pad_length >= 0:
            # BERT special token
            token = ["[CLS]"] + token + ["[SEP]"]
            pos = [0] + pos + [0]
            # padding
            token += ["[PAD]"] * pad_length
            pos += [0] * pad_length
            # dp arcs pad
            # mask pad
            mask = [1] * (token_length + 2) + [0] * pad_length
        else:
            token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
            pos = [0] + pos[:self.max_length] + [0]
            mask = [1] * (self.max_length + 2)
        # 数字化
        token = self.tokenizer.convert_tokens_to_ids(token)

        # tensor化
        if self.device is None:
            token = torch.LongTensor(token)
            pos = torch.LongTensor(pos)
            mask = torch.ByteTensor(mask)
        else:
            token = torch.LongTensor(token).cuda(self.device)
            pos = torch.LongTensor(pos).cuda(self.device)
            mask = torch.ByteTensor(mask).cuda(self.device)

        return [token, pos, mask]

    def getSentMap(self):
        return self.sent_map

    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        sent = self.datas[idx]
        # 文件中读取基础数据
        return self.loadSent(sent)

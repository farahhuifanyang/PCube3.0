'''
Author: your name
Date: 2021-04-13 09:07:21
LastEditTime: 2021-05-27 17:58:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_ore/data_reader.py
'''


from coreNLP.EL.config import FLAGS
import json
import torch
import torch.utils.data as data


class ELDataset(data.Dataset):
    def __init__(self, inputs, tokenizer, max_sen_length, max_ent_length, device=None):
        self.device = device
        self.max_sen_length = max_sen_length
        self.max_ent_length = max_ent_length
        self.tokenizer = tokenizer
        self.features = FLAGS.features
        self.neg_pos = True if FLAGS.loss != 'bce' else False
        self.inputs = inputs

    def loadSample(self, cand):
        """
        加载一条消歧操作的数据，cand包含一个wiki实体的简介

        Args:
            cand (dict): {"formal":正式名称, "exrest": 简介} 字段可能是None

        Returns:
            list: [token, desc, tok_mask, desc_mask, form, formal, ment_mask, form_mask]
        """
        token = self.inputs["sentence"]
        form = self.inputs["form"]
        formal = cand["formal"] if cand["formal"] else ""
        desc = cand["exrest"] if cand["exrest"] else ""

        # 数字化 padding 截断
        token = self.tokenizer(token, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        desc = self.tokenizer(desc, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        form = self.tokenizer(form, padding='max_length', truncation=True, max_length=self.max_ent_length, return_tensors='pt')
        formal = self.tokenizer(formal, padding='max_length', truncation=True, max_length=self.max_ent_length, return_tensors='pt')

        token, tok_mask = token["input_ids"].squeeze(), token["attention_mask"].squeeze()
        desc, desc_mask = desc["input_ids"].squeeze(), desc["attention_mask"].squeeze()
        form, ment_mask = form["input_ids"].squeeze(), form["attention_mask"].squeeze()
        formal, form_mask = formal["input_ids"].squeeze(), formal["attention_mask"].squeeze()

        # tensor化
        if self.device != -1:
            token, desc = token.cuda(self.device), desc.cuda(self.device)
            tok_mask, desc_mask = tok_mask.cuda(self.device), desc_mask.cuda(self.device)
            form, formal = form.cuda(self.device), formal.cuda(self.device)
            ment_mask, form_mask = ment_mask.cuda(self.device), form_mask.cuda(self.device)

        return [token, desc, tok_mask, desc_mask, form, formal, ment_mask, form_mask]

    def getOrigin(self, idx):
        """
        用于查看输出情况，根据id输出数据

        Args:
            idx (int): 数据在整个test_set中的索引

        Returns:
            原句，实体简介，form，正式名称
        """
        cand = self.inputs["candidates"][idx]
        return self.inputs["sentences"], cand["exrest"], self.inputs["form"], cand["formal"]

    def __len__(self):
        return len(self.inputs["candidates"])

    def __getitem__(self, idx):
        cand = self.inputs["candidates"][idx]
        # 文件中读取基础数据
        return self.loadSample(cand)

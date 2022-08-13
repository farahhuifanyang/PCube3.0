'''
Author: your name
Date: 2021-03-08 08:39:21
LastEditTime: 2021-04-30 16:29:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_project/models/ore_model.py
'''
import torch
from torch import log_softmax
import torch.nn as nn
from transformers import BertForTokenClassification
from torchcrf import CRF


class NERModel(nn.Module):
    def __init__(self, flags, bertconfig):
        super(NERModel, self).__init__()
        self.label_num = len(flags.label_map)
        bertconfig.num_labels = self.label_num
        bertconfig.return_dict = True
        bertconfig.output_hidden_states = True

        self.features = flags.features
        self.decoder = flags.decoder

        # local bert
        self.bert = BertForTokenClassification(bertconfig)
        self.bn = nn.BatchNorm1d(flags.max_length)
        self.dropout = nn.Dropout(flags.dropout_rate)

        # feature emb
        if "pos" in self.features:
            self.posEmb = nn.Embedding(len(flags.pos_map), flags.feature_dim)

        # full connection layers
        self.concat2tag = nn.Linear(
            flags.feature_dim*len(self.features)+bertconfig.hidden_size, self.label_num)
        # self.concat2tag = nn.Linear(flags.feature_dim*4, self.label_num)

        # decode layer
        if self.decoder == "crf":
            self.crf_layer = CRF(self.label_num, batch_first=True)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, token, pos, gold, mask, acc_mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(
            token, labels=gold, attention_mask=mask).hidden_states[-1]
        bert_hidden = self.bn(bert_hidden)
        # batch_size, max_length, bert_hidden
        token_emb = self.dropout(bert_hidden)

        if "pos" in self.features:
            # batch_size, max_length, feature_dim
            pos_emb = self.posEmb(pos)

        # feature concat, fc layer
        logits = token_emb
        if "pos" in self.features:
            logits = torch.cat([logits, pos_emb], dim=-1)
        # batch_size, max_length, bert_hidden + feature_dim
        logits = self.concat2tag(logits)

        if self.decoder == "crf":
            # crf loss
            loss = - self.crf_layer(logits, gold,
                                    mask=acc_mask, reduction="mean")
            loss += - self.crf_layer(logits, gold, mask=mask, reduction="mean")
            pred = torch.Tensor(self.crf_layer.decode(logits)).cuda()
        else:
            # softmax loss
            loss = self.loss(logits.view(-1, self.label_num), gold.view(-1))
            pred = torch.max(log_softmax(logits, dim=-1), dim=-1).indices

        zero = torch.zeros(*gold.shape, dtype=gold.dtype).cuda()
        eq = torch.eq(pred, gold.float())
        acc = torch.sum(eq * acc_mask.float()) / torch.sum(acc_mask.float())
        zero_acc = torch.sum(torch.eq(zero, gold.float())
                             * mask.float()) / torch.sum(mask.float())

        return loss, acc, zero_acc

    def decode(self, token, pos, mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(
            token, attention_mask=mask).hidden_states[-1]
        bert_hidden = self.bn(bert_hidden)
        # batch_size, max_length, bert_hidden
        token_emb = self.dropout(bert_hidden)

        if "pos" in self.features:
            pos_emb = self.posEmb(pos)

        # feature concat, fc layer
        logits = token_emb
        if "pos" in self.features:
            logits = torch.cat([logits, pos_emb], dim=-1)
        logits = self.concat2tag(logits)

        if self.decoder == "crf":
            # crf decode
            tag_seq = self.crf_layer.decode(logits, mask=mask)
        else:
            # softmax decode
            tag_seq = torch.max(log_softmax(logits, dim=-1), dim=-1).indices
            tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq

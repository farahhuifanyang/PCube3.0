'''
Author: Liuhezi
Date: 2021-04-22 11:02:31
LastEditTime: 2021-04-27 19:51:50
LastEditors: Liuhezi
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/PER/model.py
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

torch.set_default_tensor_type(torch.DoubleTensor)

class PERModel(nn.Module):
    
    def __init__(self, pern_feature, word_feature,
                 pern_adj, word_pern_adj,
                 n_units=None, n_heads=None, user_dim=256,
                 dropout=0.1, attn_dropout=0):
        super(PERModel, self).__init__()
        if n_units is None:
            n_units = [32, 32]
        if n_heads is None:
            n_heads = [8, 8, 1]
        self.dropout = dropout
        n_units = n_units + [user_dim]
        self.pern_feature = pern_feature
        self.word_feature = nn.Parameter(word_feature)
        self.pern_adj = pern_adj
        self.word_pern_adj = word_pern_adj
        self.gat1_pp = MultiHeadGraphAttention(n_heads[0], f_in=n_units[0] * n_heads[0],
                                               f_out=n_units[1], attn_dropout=attn_dropout)

        self.gat2_wp = MultiHeadGraphAttention(n_heads[1], f_in=n_units[1] * n_heads[1],
                                               f_out=n_units[2], attn_dropout=attn_dropout)

        self.fc = nn.Linear(n_units[-1], 5)

    def forward(self,input):
        n_p = self.pern_feature.shape[0]
        hiddn_pp = self.gat1_pp(self.pern_feature,self.pern_adj)
        hiddn_pp = F.elu(hiddn_pp.transpose(0,1).contiguous().view(n_p,-1))
        hiddn_pp = F.dropout(hiddn_pp,self.dropout,training=self.training)

        hiddn_wp = torch.cat((self.word_feature,hiddn_pp),0)
        hiddn_wp = self.gat2_wp(hiddn_wp,self.word_pern_adj)
        hiddn_wp = hiddn_wp.mean(dim=0)

        hiddn_w = hiddn_wp[:self.word_feature.size(0),:]

        x = torch.div(torch.mm(input,hiddn_w),input.sum(1).unsqueeze(1))
        output = self.fc(x)
        return output


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # adj: n x n
        # h: n x f_in
        n = h.size(0)
        # n x f_in
        h_prime = torch.matmul(h, self.w)
        # n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)
        # n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)
        # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + \
               attn_dst.expand(-1, -1, n).permute(0, 2, 1)
        # n_head x n x n
        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(0)
        # 1 x n x n
        attn.data.masked_fill_(mask.bool(), float("-inf"))
        attn = self.softmax(attn)
        # n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        # n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output
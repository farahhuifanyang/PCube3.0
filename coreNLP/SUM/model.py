'''
Author: Guowenying
Date: 2021-04-28 12:17:48
LastEditTime: 2021-06-03 11:18:25
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/SUM/model.py
'''

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import AlbertConfig, AlbertModel

class AlbertAttn(nn.Module):
    def __init__(self):
        super(AlbertAttn, self).__init__()

        Albert_config=AlbertConfig.from_pretrained('voidful/albert_chinese_base')
        self.model = AlbertModel(Albert_config)
        self.model.config.output_hidden_states = True

    def forward(self, x, segs, mask):
        """
        将文章输入到ALBERT模型中，获得每一token的输出表示及其隐层的表示

        Parameters
        ----------
        x : tensor
            文章中的token在vocab中的index
        segs : tensor
            指示该token是在奇数句还是在偶数句中。表示段信息
        mask : tensor
            如果文章的长度不同，标记该位置是否被padding

        Returns
        -------
        [type]
            [description]
        """
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        all_layer = torch.cat((
                             output[2][1].unsqueeze(dim=0), output[2][2].unsqueeze(dim=0),
                             output[2][3].unsqueeze(dim=0), output[2][4].unsqueeze(dim=0),
                             output[2][5].unsqueeze(dim=0), output[2][6].unsqueeze(dim=0),
                             output[2][7].unsqueeze(dim=0), output[2][8].unsqueeze(dim=0),
                             output[2][9].unsqueeze(dim=0), output[2][10].unsqueeze(dim=0),
                             output[2][11].unsqueeze(dim=0), output[2][12].unsqueeze(dim=0)
        ), dim=0).transpose(0,1)
        return output[0], all_layer


class AlbertSummarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, Albert_config = None):
        super(AlbertSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = AlbertAttn()
        self.encoder = RNNEncoderAttn(bidirectional=True, num_layers=1,
                                    input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                    dropout=args.dropout)
        self.score = nn.Linear(self.bert.model.config.hidden_size*2,1,bias=True)
        self.sigmoid = nn.Sigmoid()

        self.to(device)
    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path), strict=True)

    def forward(self, x, segs, clss, mask, mask_cls):
        """
        文本摘要模型， 具体实现可参考-Complementary Representation of ALBERT for Text Summarization
        Parameters
        ----------
        x : tensor
           文档中的单词在vocab中的idx
        segs : tensor
            段信息-指示当前词是在奇数句还是偶数句中
        clss : tensor
            指示每一个[CLS]在文章中的位置，便于后续提取句子的embedding
        mask : tensor
            指示某些位置是否被padding
        mask_cls : tensor
            clss的mask, 因为每篇文章的句子个数是不同的

        Returns
        -------
        tensor; tensor
            每个句子的得分; 指示当前句是不是被mask， 若是该句得分归零
        """
        last, top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec.transpose(1, 2)[torch.arange(top_vec.size(0)).unsqueeze(1), clss].transpose(1, 2)
        last = last[torch.arange(last.size(0)).unsqueeze(1), clss]
        batch, layer, seq, hidden = top_vec.size()
        
        sents_vec = self.encoder(sents_vec)  # encoder应该加入mask_cls
        sents_vec = sents_vec.view(batch, -1, hidden)
        
        sents_vec = torch.cat((sents_vec, last), dim=-1)  # 第一次效果有提升
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        
        score = self.score(sents_vec)
        sent_scores = self.sigmoid(score) * mask_cls[:, :, None].float()
        return sent_scores.squeeze(-1), mask_cls

class RNNEncoderAttn(nn.Module):

    def __init__(self, bidirectional, num_layers, input_size,
                 hidden_size, dropout=0.0):
        super(RNNEncoderAttn, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.relu = nn.ReLU()

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional)

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        将获得的ALBERT每一层的隐层输出输入到该模块中，获得每一隐层的分布式表示，用于计算该层的重要性，然后获得ALBERT最终输出的补充表示

        Parameters
        ----------
        x : tensor
            ALBERT每一层的隐层输出

        Returns
        -------
        tensor
            ALBERT最终输出的补充表示
        """
        batch, layer, seq, hidden = x.size()
        x1=x.contiguous().view(batch * layer, -1, hidden)
        x1 = torch.transpose(x1, 1, 0)
        memory_bank, _ = self.rnn(x1)
        memory_bank = self.dropout(memory_bank) + x1
        memory_bank = torch.transpose(memory_bank, 1, 0)

        sent_scores = self.softmax(self.relu(self.wo(memory_bank[:,-1,:])).squeeze(dim=-1).view(-1,layer)).unsqueeze(-1)
        x=x.transpose(1,2)
        sent_vec = torch.matmul(sent_scores.transpose(1,2).unsqueeze(dim = 1).expand(batch,seq,1,layer),x)

        return sent_vec.squeeze(dim = 2)

class LayerNormLSTMCell(nn.LSTMCell):
    """
    修改LSTM中cell的计算方式

    Parameters
    ----------
    nn : class
        torch.nn模块
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        """
        在LSTM中实现Layer Normalization. 在计算输入门，输出门，遗忘门加入Layer Normalization操作

        Parameters
        ----------
        input : tensor
            LSTM当前步的输入
        hidden : tensor
            上一步的状态

        Returns
        -------
        tensor
            LSTM的输出和cell状态
        """
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class LayerNormLSTM(nn.Module):
    """
    构建包含 Layer Normalization 操作的 Bi-LSTM，可以保持训练的稳定性

    Parameters
    ----------
    nn : class
        torch.nn模块
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, bias=bias)
            for layer in range(num_layers)
        ])

        if self.bidirectional:  #是否是双向LSTM
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, bias=bias)
                for layer in range(num_layers)
            ])

    def forward(self, input, hidden=None):
        """
        包含 Layer Normalization 操作的 LSTM的具体实现

        Parameters
        ----------
        input : tensor
            模型的输入
        hidden : tensor, optional
            LSTM模型的隐层状态, by default None

        Returns
        -------
        tensor, (tensor, tensor)
            LSTM 的输出, (LSTM 的隐层状态, LSTM 的cell状态)
        """
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.num_layers * num_directions)] * seq_len
        ct = [[None, ] * (self.num_layers * num_directions)] * seq_len

        if self.bidirectional:
            xs = input
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])

        return y, (hy, cy)

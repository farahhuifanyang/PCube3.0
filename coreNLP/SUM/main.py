'''
Author: Guowenying
Date: 2021-04-28 11:52:10
LastEditTime: 2021-06-09 23:32:35
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/SUM/main.py
'''

import torch
from coreNLP.SUM.config import args
from coreNLP.SUM import DataParser
from coreNLP.SUM import model 
from coreNLP.SUM.DataParser import loadDataset
from coreNLP.Algorithm import Algorithm
from configure import globalFLAGS
from coreNLP.SUM.model import AlbertSummarizer
import re
import datetime
import jieba
import numpy as np
from DAO.ElasticFile import ElasticFile

# 预测时需调整的模型参数
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder', 'use_interval','rnn_size']


class ExtSummarization(Algorithm):
    def __init__(self):
        if globalFLAGS.SUM_cuda_visible_devices != -1:
            self.device = globalFLAGS.SUM_cuda_visible_devices
        else:
            self.device = None
        
    def run(self):
        """
        加载模型并抽取出文本摘要

        Returns
        -------
        dict
            {事件ID:摘要}
        """

        # es.indices.create(index='event_summary')
        es = ElasticFile(globalFLAGS.ES_url)
        self.model = AlbertSummarizer(args, self.device)
        self.model.load(globalFLAGS.SUM_model_path)
        self.model.eval()
        self.loss = torch.nn.BCELoss(reduction='none')
        if self.device:
            self.model.cuda(self.device)
        num = 0
        test_iter =DataParser.SUMDataset(args, loadDataset(args, 'test'), args.batch_size, self.device)
        def clean(s):
            """
            将字符串中的标点符号清除
            Parameters
            ----------
            s : string
                待清除的字符串

            Returns
            -------
            string
                返回清除后的字符串
            """
            return re.sub(r'[0-9a-zA-Z.， ,\-。%《*》/•、&＆(—)（+）：？!！“”·:]', '', s)#\u4e00-\u9fa5
        def _get_Ngrams(n, text):
            """
            获得句子的n-gram集合
            Parameters
            ----------
            n : int
                n的取值代表n-gram
            text : string
                待处理的句子

            Returns
            -------
            set
                获得的句子的n-gram集合
            """
            ngram_set = set()
            for sentence in text:
                texts = jieba.cut(sentence,cut_all=True)
                word = ('/'.join(texts)).split('/')
            
                text_length = len(word)
                max_index_ngram_start = text_length - n
                for i in range(max_index_ngram_start + 1):
                    ngram_set.add(tuple(word[i:i + n]))
            # text_length = len(text)
            # max_index_ngram_start = text_length - n
            # for i in range(max_index_ngram_start + 1):
            #     ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_Tri(c, p, n):
            """
            摘要中的Trigram Blocking方法，有效的去除冗余，判断候选句与已有选中的摘要句是否有n-gram重叠
            Parameters
            ----------
            c : string
                已经被选中的候选摘要
            p : string
                当前预测的摘要句

            Returns
            -------
            Boolean
                指示该句是否应该在摘要集合中
            """
            tri_c = _get_Ngrams(n, c.split())
            for s in p:
                tri_s = _get_Ngrams(n, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False
        summary = {}
        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                selected_ids = np.argsort(-sent_scores, 1)
                for i, idx in enumerate(selected_ids):
                    _pred = []
                    _cand = []
                    if(len(batch.src_str[i])==0):
                        continue
                    for j in selected_ids[i][:len(batch.src_str[i])]:
                        if(j>=len( batch.src_str[i])):
                            continue

                        words = list(clean(batch.src_str[i][j].strip()))
                        candidate =' '.join(words) 
                        if(args.block_trigram):
                            if((not _block_Tri(candidate,_pred,7)) and (_block_Tri(candidate,_pred,2))) or len(_pred)==0:
                                _pred.append(candidate)
                                _cand.append(batch.src_str[i][j].strip())
                        else:
                            _pred.append(candidate)
                            _cand.append(batch.src_str[i][j].strip())

                        if len(_pred) == 3:
                            break

                    _cand = ''.join(_cand)  # 文章摘要的id表示，用于计算ROUGE比分
                    summary[batch.tgt_str[i]] = _cand
                    num += 1
                    if num%5000==0:
                        print(num)
                    result = es.searchByKey(key = batch.tgt_str[i], index = 'event_summary')
                    if result is None or len(result['name']) ==0:
                        es.write(index="event_summary",doc_type="doc",id=batch.tgt_str[i],data={"eid":batch.tgt_str[i], "abstract":summary[batch.tgt_str[i]], "name":"","timestamp":datetime.datetime.now()}) 
                    # else:
                    #     es.write(index="event_summary",doc_type="doc",id=batch.tgt_str[i],body={"eid":batch.tgt_str[i], "abstract":summary[batch.tgt_str[i]], "name":"","timestamp":datetime.datetime.now()}) 
        print(num)
        return summary

if __name__ == '__main__':
    model = ExtSummarization()
    model.run()
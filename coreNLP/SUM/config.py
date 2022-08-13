'''
Author: Guowenying
Date: 2021-04-26 12:23:55
LastEditTime: 2021-06-03 11:36:36
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/SUM/config.py
'''
import os
import argparse
from pathlib import Path
class Flags(object):
    def __init__(self):

        #训练模块参数
        self.bert_data_path = '/home/disk2/nuclear/PCube_tmp/SUM/albert_data/events'    # 需要读取的tensor文件路径
        self.test_from = Path('models/SUM/model_step_5000.pt')
        self.head = 4
        self.inter_layers = 1
        self.batch_size = 3000
        self.use_interval = True
        self.hidden_size = 768
        self.ff_size = 512
        self.rnn_size = 768
        self.dropout = 0.1
        self.optim = 'adam'
        self.world_size = 1
        self.warmup = 15000
        self.lr =  2e-3
        self.decay_method = 'noam'
        self.warmup_steps = 10000
        self.block_trigram = True   

        #数据预处理模块参数

        self.map_path = '/home/disk2/nuclear/PCube_tmp/CLUSTER/clusterTopic.txt'           #存储事件和包含的文章映射的文件
        self.raw_path = '/home/disk2/nuclear/PCube_tmp/SUM/source_data/'       #存储生成的原始事件文件夹
        self.save_path = '/home/disk2/nuclear/PCube_tmp/SUM/target_data/'      #存储分词后的事件文件夹
        self.json_read_path = '/home/disk2/nuclear/PCube_tmp/SUM/json_data/'   #读取json格式的事件
        self.json_path = '/home/disk2/nuclear/PCube_tmp/SUM/json_data/events'  #处理后的事件存储成json格式
        self.bert_path = '/home/disk2/nuclear/PCube_tmp/SUM/albert_data/'      #存储转化成albert模型能够读取的张量格式
        self.source_path = '/home/gwy/taihai/政治'                              #存储最初的原始文件 article
        
        self.shard_size = 2000      #一个文件能够包含的事件的个数
        self.min_nsents = 3         #文章中至少包含多少个句子
        self.max_nsents = 100       #文章中至多包含多少个句子
        self.min_src_ntokens = 5    #一个句子至少包含多少个字
        self.max_src_ntokens = 200  #一个句子至多包含多少个字
        self.n_cpus = 2             #并行处理指定的线程个数


args = Flags()
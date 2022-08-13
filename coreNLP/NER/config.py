'''
Author: your name
Date: 2020-09-23 09:23:31
LastEditTime: 2021-05-11 11:32:01
LastEditors: Please set LastEditors
Description: code and model configs
FilePath: /entity_disambiguation/config.py
'''
import os
import json
import time
from configure import globalFLAGS


class Flags(object):
    def __init__(self):
        # task info
        self.is_test = True
        self.canonicalizing = False

        # data dirs
        curpath = os.path.abspath(os.path.dirname(__file__))
        self.pretrained = "bert-base-chinese"

        # Path of output results dir
        self.out_dir = os.path.join(curpath, "out")
        self.record_path = os.path.join(self.out_dir, "ner_record.txt")

        # train hyper parameters
        self.learning_rate = 3.e-5
        self.epoch = 100
        self.batch_size = 64
        # self.test_batch_size = 1
        self.max_length = 128
        self.dropout_rate = 0.5
        self.weight_decay = 1.e-3
        self.patient = 3
        self.use_cuda = True

        # TransD config
        self.trans_select = "TransD"  # TransE TransH
        self.feature_dim = 100
        self.margin = 4.0

        # model choice
        self.features = []   # ["dp", "pos", "ner"]
        self.decoder = "crf"  # crf, softmax

        # lstm
        self.lstm_hidden = 300
        self.n_layers = 2

        # global datas
        self.label_map = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
                          "B-LOC": 5, "I-LOC": 6, "B-REG": 7, "I-REG": 8, "B-OTH": 9, "I-OTH": 10}
        self.id2label = {0: "O", 1:  "B-PER", 2:  "I-PER", 3:  "B-ORG", 4:  "I-ORG", 5:
                         "B-LOC", 6:  "I-LOC", 7:  "B-REG", 8:  "I-REG", 9:  "B-OTH", 10: "I-OTH"}
        # self.dp_map = json.load(
        #     open(os.path.join(self.data_dir, "dp_map.json")))
        self.pos_map = json.load(open(globalFLAGS.NER_pos_path))


FLAGS = Flags()

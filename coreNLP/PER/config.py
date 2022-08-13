'''
Author: Liuhezi
Date: 2021-04-22 09:57:02
LastEditTime: 2021-04-26 11:25:50
LastEditors: Liuhezi
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/PER/config.py
'''
import argparse


class Flags(object):
    def __init__(self):
        self.epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 5e-4
        self.dropout = 0.2
        self.hidden_units = "32,32"
        self.feature_size = "32,256"
        self.heads = "8,8,1"
        self.batch_size = 64
        self.patience = 20
        self.save_dir = ""
        self.model_path = ""


FLAGS = Flags()
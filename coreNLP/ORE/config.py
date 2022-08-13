'''
Author: Huifan Yang
Date: 2021-06-02
LastEditTime: 2021-06-05 11:23:11
LastEditors: Please set LastEditors
Description: Define configuration
FilePath: /PCube3.0/coreNLP/ORE/config.py
'''
import os
import json
import time
import torch
from configure import globalFLAGS


class Flags(object):
    def __init__(self):
        curpath = os.path.abspath(os.path.dirname(__file__))
        self.json_folder_path = globalFLAGS.ORE_input_folder
        self.state_dict = globalFLAGS.ORE_model_path
        # Path of output results dir
        self.output_dir = globalFLAGS.ORE_output_path
        self.ORE_output_path = ""

        self.model_type = "bert"
        self.model_name_or_path = "bert-base-chinese"
        self.no_cuda = False
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = None
        self.do_train = False
        self.do_eval = True
        self.logging_steps = 50
        self.overwrite_cache = True
        self.local_rank = -1
        self.n_gpu = torch.cuda.device_count()

        self.max_seq_length = 384
        self.max_query_length = 64
        self.doc_stride = 128
        self.per_gpu_eval_batch_size = 8

        self.version_2_with_negative = True
        self.eval_all_checkpoints = True
        self.n_best_size = 20
        self.max_answer_length = 30
        self.do_lower_case = False
        self.verbose_logging = True
        self.null_score_diff_threshold = 0.0


FLAGS = Flags()

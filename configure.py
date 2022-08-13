'''
Author: your name
Date: 2020-12-29 14:22:08
LastEditTime: 2021-06-22 11:01:33
LastEditors: Please set LastEditors
Description: Configurations
FilePath: /PCube3/configure.py
'''

import os


class Flags(object):
    def __init__(self):
        # public data dir
        curpath = os.path.abspath(os.path.dirname(__file__))
        self.model_dir = os.path.join(curpath, "models")  # Path of model checkpoints dir 请不要改变该项目
        self.static_dir = os.path.join(curpath, "static")  # Path of input data dir 请不要改变该项目
        self.log_dir = os.path.join(curpath, "logs")
        self.out_dir = os.path.join(curpath, "out")  # type=str, Path of output results dir 请不要改变该项目

        # News sites crawl configs
        self.news_data_dir = "/home/disk2/nuclear/news_data/PCube"
        self.tmp_result_dir = "/home/disk2/nuclear/PCube_tmp/"
        self.picture_dir = "/home/disk2/nuclear/PCube_pic/"
        self.chinatimes_dir = 'chinatimes'

        # DBMS configs
        self.Hbase_ip = "10.105.242.73"
        self.Hbase_prefix = "PCube"
        self.ES_url = "10.105.242.74:9200"
        self.ES_news_dir = "/PCube"
        self.ES_event_dir = "/PCube"
        self.neo4j_url = "bolt://10.105.242.74:7687"  # neo4j服务器的url
        self.neo4j_usr = "neo4j"  # neo4j用户名
        self.neo4j_passwd = "neo4j"  # neo4j密码

        # preprocess configs

        # 建议GPU的配置尽可能不同，以在流式执行中能够取得类似流水线的效果
        # entity recognition configs     请根据实际需要配置
        self.NER_model_path = 'models/NER/04-12-14.pkl'  # NER模型的存放位置
        self.NER_pos_path = 'models/NER/pos_map.json'  # NER使用的pos映射表的存放位置
        self.NER_cuda_visible_devices = 0      # 配置GPU设备号 -1 表示使用CPU 下同
        self.NER_batch_size = 90    # 应该根据模型本身的显存占用量和数据的占用量综合计算

        # entity linking configs     请根据实际需要配置
        self.EL_model_path = 'models/EL/log_best3.pkl'  # EL模型的存放位置
        self.EL_cuda_visible_devices = 0
        self.EL_candi_limit = 10
        self.EL_proxy_port = 7890
        self.threshold = 0.5

        # open relation extraction configs 请根据实际需要配置
        self.ORE_model_path = 'models/ORE/pytorch_model.bin'
        self.ORE_cuda_visible_devices = 0
        self.ORE_output_path = 'models/ORE/output'
        self.ORE_predict_file = 'models/ORE/input.json'
        self.ORE_input_folder = '/home/disk2/nuclear/PCube_tmp/EL_Linked'

        # Abstractions configs  请根据实际需要配置
        self.SUM_model_path = 'models/SUM/21-06-03.pkl'
        self.SUM_cuda_visible_devices = 1
        self.SUM_batch_size = 30

        # personality configs  请根据实际需要配置
        self.PER_model_path = 'models/PER/04-19-12.pkl'
        self.PER_pern_features_path = 'models/PER/pern_feature_256.npy'
        self.PER_pern_adj_path = 'models/PER/pern_pern_adj.npy'
        self.PER_word_pern_adj_path = 'models/PER/word_pern_adj.npy'
        self.PER_word_list_path = 'models/PER/word_list.npy'
        self.PER_tw_liwc_dict_path = 'models/PER/tw_liwc_dict.json'
        self.PER_liwc_mean_path = 'models/PER/liwc_mean.npy'
        self.PER_entity_path = 'models/PER/entity.pkl'
        self.PER_user_id_url = 'models/PER/FB_id_url.txt'
        self.PER_cuda_visible_devices = 1
        self.PER_batch_size = 30

        # sentiment analysis configs  请根据实际需要配置
        self.SA_model_path = '.pkl'
        self.SA_cuda_visible_devices = 0
        self.SA_batch_size = 30


globalFLAGS = Flags()

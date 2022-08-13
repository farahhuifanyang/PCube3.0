'''
Author: Guowenying
Date: 2021-05-11 19:35:37
LastEditTime: 2021-06-08 20:54:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/CLUSTER/config.py
'''


import os
import argparse
from pathlib import Path
from configure import globalFLAGS


class Flags(object):
    def __init__(self):
        #task info
        self.stop_words_file = 'static/preproces/stop_word.txt'  # 停用词文件位置
        self.temp_file = os.path.join(globalFLAGS.tmp_result_dir, 'CLUSTER')   # 文件夹存放临时文件
        self.number = 1000   #处理的文件个数， -1代表全部处理
        self.theta = 0.1   #设置相似度阈值
        self.ltp_data_dir='/home/nuclear/ltp_data_v3.4.0/'
        self.dir_name = globalFLAGS.news_data_dir
        self.file_suffix = 'html'


args = Flags()

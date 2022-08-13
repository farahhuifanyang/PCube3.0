'''
Author: your name
Date: 2021-04-06 14:40:12
LastEditTime: 2021-04-06 17:35:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/Algorithm.py
'''


class Algorithm:
    def __init__(self) -> None:
        """
        加载模型设置
        加载模型参数
        将模型放入GPU

        Raises:
            NotImplementedError: if not implemented by children class
        """
        raise NotImplementedError

    def run(self, **inputs):
        """
        使用输入构造一批数据的torch DataLoader
        以mini_batch为单位得到预测结果
        根据预测结果还原出模型的最终结果

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    '''
    # 建议应该有的私有方法
    def postProcess(preds):
        """
        根据模型的直接输出构造算法最终输出
        其间有必要针对模型容易犯的错误进行特殊处理，必要时放弃召回率提升准确率

        Args:
            preds (Tensor): 模型的直接输出

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError
    '''

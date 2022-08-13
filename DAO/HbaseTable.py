'''
Author: your name
Date: 2021-05-14 10:54:43
LastEditTime: 2021-05-14 15:30:22
LastEditors: Please set LastEditors
Description: 定义保存在Hbase中的表格的基类
FilePath: /PCube3.0/DAO/HbaseTable.py
'''
import happybase


class HbaseTable(object):
    def __init__(self, host, prefix, table_name) -> None:
        self.connection = happybase.Connection(host=host, table_prefix=prefix)
        self.table = self.connection.table(table_name)

    def searchByKey(self, key):
        """
        在表格中以主键或外键进行精确搜索
        返回匹配成功的行

        Args:
            key (str): 主键或外键

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def searchByValue(self, filter):
        """
        在表格中以列的值进行筛选性搜索
        返回匹配成功的行

        Args:
            filter (dict): 列-值的对应表，格式！！！暂定！！！为{"列名": "列值"}

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def write(self, rowkey, data):
        """
        向表格中新增一条数据

        Args:
            rowkey (str): 写入数据的主键值
            data (dict)): 以col_name, value组织的新数据，格式为{"列名": "列值"}

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

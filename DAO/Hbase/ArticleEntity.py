'''
Author: your name
Date: 2021-06-09 14:15:59
LastEditTime: 2021-06-09 14:54:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/DAO/Hbase/EntityArtical.py
'''
import happybase
from DAO.HbaseTable import HbaseTable


class ArticleEntity(HbaseTable):
    def __init__(self, host, prefix, table_name="ArticleEntity") -> None:
        super().__init__(host, prefix, table_name)
        self.host = host
        self.prefix = prefix
        self.table_name = table_name

    def searchByAID(self, aid):
        """
        在表格中以主键或外键进行精确搜索
        返回匹配成功的行

        Args:
            aid (str): 文章的aid

        Return:
            list[dict]: 返回匹配成功的行
        """
        self.refresh()
        key = aid.encode()
        rows = self.table.scan(row_prefix=key)
        res_rows = []
        for row in rows:
            row_dict = {}
            row = row[1]
            for k, v in row.items():  # 二进制转换
                k = k.decode().split(":")[-1]
                v = v.decode()
                v = v.replace("%20", " ")
                row_dict[k] = v
            res_rows.append(row_dict)
        return res_rows

    def searchByEID(self, eid):
        """
        在表格中以列的值进行筛选性搜索
        返回匹配成功的行

        Args:
            eid (str): 外键实体的id

        Raises:
            NotImplementedError: [description]
        """
        self.refresh()
        filter = f"SingleColumnValueFilter('eid', 'eid', =, 'binary:{eid}')"
        rows = self.table.scan(filter=filter)
        res_rows = []
        for row in rows:
            row_dict = {}
            row = row[1]
            for k, v in row.items():  # 二进制转换
                k = k.decode().split(":")[-1]
                v = v.decode()
                v = v.replace("%20", " ")
                row_dict[k] = v
            res_rows.append(row_dict)
        return res_rows

    def write(self, aid, eid):
        """
        向表格中新增一条数据

        Args:
            rowkey (str): 写入数据的主键值
            data (dict)): 以col_name, value组织的新数据，格式为{"列名": "列值"}

        Raises:
            NotImplementedError: [description]
        """
        self.refresh()
        put_data = {"eid:eid": eid}
        rowkey = aid + "_" + eid
        self.table.put(rowkey, put_data)

    def refresh(self):
        self.connection = happybase.Connection(host=self.host, table_prefix=self.prefix)
        self.table = self.connection.table(self.table_name)


if __name__ == "__main__":
    eatable = ArticleEntity("10.105.242.73", "PCube")
    aid = "7a56ab4c02341a698201510f39cc5732"
    eatable.searchByAID(aid)
    eid = "Q715869"
    eatable.searchByEID(eid)

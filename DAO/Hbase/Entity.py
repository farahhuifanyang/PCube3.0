'''
Author: your name
Date: 2021-05-14 10:54:22
LastEditTime: 2021-06-09 09:41:37
LastEditors: Please set LastEditors
Description: 定义实体歧义表格的读写方法
FilePath: /PCube3.0/DAO/Entity.py
'''
import happybase
from DAO.HbaseTable import HbaseTable


class Entity(HbaseTable):
    def __init__(self, host, prefix, table_name="Entity") -> None:
        super().__init__(host, prefix, table_name)
        self.host = host
        self.prefix = prefix
        self.table_name = table_name

    def searchByKey(self, key, prefix=False):
        """
        在表格中以主键或外键进行精确搜索
        返回匹配成功的行    未完成！！！

        Args:
            key (str): 主键或外键的值
            prefix (bool): 为真时用key作为prefix匹配，而非直接用来匹配主键

        Return:
            list[dict]: 返回匹配成功的行
        """
        self.refresh()
        if not prefix:
            row_dict = {}
            row = self.table.row(key)
            for k, v in row.items():  # 二进制转换
                k = k.decode().split(":")[-1]
                v = v.decode()
                v = v.replace("%20", " ")
                row_dict[k] = v
            if row_dict:
                return [row_dict]
        else:
            key = key.encode()
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

    def searchByValue(self, filter):
        """
        在表格中以列的值进行筛选性搜索
        返回匹配成功的行    未完成！！！

        Args:
            filter (dict): 列-值的对应表，格式为{"列名": "列值"}

        Raises:
            NotImplementedError: [description]
        """
        self.refresh()
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
        self.refresh()
        put_data = {}
        for k, v in data.items():
            if k in ["introduction", "webSite"]:
                k = "common:"+k
            else:
                k = "properties:"+k
            put_data[k] = v
        self.table.put(rowkey, put_data)

    def refresh(self):
        """
        由于Thrift的连接过了一定时间就会断掉，需要在使用前重置链接
        """
        self.connection = happybase.Connection(host=self.host, table_prefix=self.prefix)
        self.table = self.connection.table(self.table_name)


if __name__ == "__main__":
    import json
    f = "static/processOnBaseData/criticalEntity.json"
    table = Entity("10.105.242.73", "PCube")
    with open(f, 'r') as cef:
        datas = json.load(cef)
        for i, data in enumerate(datas):
            rowkey = data["id"]
            res = table.searchByKey(rowkey, prefix=True)[0]
            intro = res["introduction"]
            datas[i]["introduction"] = intro
    with open(f, 'w') as cef:
        json.dump(datas, cef, ensure_ascii=False)

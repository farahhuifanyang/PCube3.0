'''
Author: Guowenying
Date: 2021-06-03 09:25:28
LastEditTime: 2021-06-08 17:42:38
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/CLUSTER/hbaseWriter.py
'''
import happybase
import json



connection = happybase.Connection(host='10.105.242.73',table_prefix='PCube',port=9090)
connection.open()

# 创建表
connection.create_table(
   'Event',
   {
   'Event_details':dict()
   }
)

#删除表
# connection.delete_table('Event',disable=True)

# 全局查询
# num = 0
# table = connection.table('Event')
# for key, value in table.scan():
#     # print(key,value)
#     num+=1
# print(num)

# 查表
#table_name_list = connection.tables()
#print(table_name_list)

# 存储
table = connection.table('Event')
bat = table.batch()

with open('/home/disk2/nuclear/PCube_tmp/CLUSTER/clusterTopic.txt','r') as f:
	mapping = json.load(f)
num = 0
for eventID, articleIDs in mapping.items():
        data = {
            'Event_details:aid':','.join(articleIDs),
        }
        num += 1
        bat.put(eventID,data)
        if num > 0 and num % 10 == 0:
            bat.send()
bat.send()
connection.close()

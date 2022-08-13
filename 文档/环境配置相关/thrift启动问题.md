# thrift启动问题

## 第一次问题

重启hbase后thrift服务未启动，使用netstat -nl | grep 9090查看9090端口无服务。

ssh连接到主节点cdh01，使用hbase thrift start命令启动thrift服务，此时测试可以通过python连接hbase，关闭ssh会话后无法继续链接，9090端口无服务。

得出结论，使用hbase thrift start开启服务不能关闭会话。

解决思路一：在cdh01上安装screen，使用screen创建窗口，执行命令hbase thrift start关闭窗口至后台运行。

解决思路二：在cdh01开机脚本中写入hbase thrift start命令，父进程为操作系统，不会被关闭会话。（未测试）

解决思路三：使用hbase-deamon.sh start thrift命令。（未测试）

使用思路一解决问题，测试可以使用python连接hbase。

## 第二次问题

配置hue可视化hbase后，可能更改了某些配置，导致thrift服务异常。

查看运行thrift服务的screen异常关闭，使用hbase thrift start重新开启时报错提示端口占用。

进入cdh集群web管理端，查看hbase的thrift相关配置，发现“启用HBase Thrift Http服务器”和“启用HBase Thrift代理用户”选项被勾选，原本是未勾选状态，取消勾选后重启相关服务（hbase、spark、hue等）。

重启后测试可以通过python连接hbase。（此时没有在screen中运行hbase thrift start）
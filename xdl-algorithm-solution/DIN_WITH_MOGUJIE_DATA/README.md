####背景

1.把当前算法线上wide&deep的模型的其中的deep部分，改造成xdl的版本；
2.然后使用蘑菇街wide&deep模型的真实样本数据来训练；
3.以此对比：在实际场景下，xdl版本和tf版本的训练速度、资源消耗、收敛情况、auc等；

####现有模型

1.git链接

     http://gitlab.mogujie.org/yangming/rank_tf/blob/distribute_dixin_tfrank_adam_v1/models/wd/wide_deep_attention_loss.py

2.tiny+配置：

    http://tiny.meili-inc.com/appDetailDefineV2?appId=557

3.样本地址：

     hdfs://mgjcluster/user/dixin/deep/wide_deep_base/2019-02-11/

4.TODO：分布式训练benchmark


#### xdl 版本改造思路

1.样本读取使用tf的data api；再通过xdl.py_func定义op，再转换成xdl的数据格式

2.使用xdl的api构建模型

3.单机测试 & 分布式测试 & timeline分析

4.现有模型同等资源配置和模型结构下跑测试，进行对比




# 成员推理攻击
成员攻击方式出自“When Machine Unlearning Jeopardizes Privacy”[1] 

参考代码：https://github.com/MinChen00/UnlearningLeaks/tree/main

[1] M. Chen, Z. Zhang, T. Wang, M. Backes, M. Humbert, Y. Zhang, When Machine Unlearning Jeopardizes Privacy, in: Acm Conference On Computer And Communications Security, 2021: pp. 896–911. https://doi.org/10.1145/3460120.3484756.

# 模拟联邦学习
该代码在[1]的基础上模型训练的方式使用模拟联邦学习的框架，遗忘的方式使用的是去掉某个客户端然后重新训练

# 运行
1. 在 GetData.py 中修改dataset_name
   执行 GetData.py (mnist可以直接执行，其他需要先获取数据保存在data文件夹下),
   处理获得模拟多个客户端的数据

2. 执行main.py中
   
    ModelTrainer(args)  执行原始模型和对比模型训练
   
    AttackModelTrainer(args)    训练攻击模型

# 文件介绍
log  日志

model/数据集名字/原始模型名字 是训练的影子模型

训练数据在data/slice

参数设置在argument.py

import numpy as np
import torch
from functools import reduce
import operator
from Model import LeNet
from Traning import ClientTrainer
from multiprocessing import Pool
from SecAgg import SecAgg


def ClientTraining(args):
    # 读取参数
    client = args["client"]
    local_epoch = args["local epoch"]
    # 读取训练和测试数据集
    global_parameters_path = args["global parameter path"]
    global_parameters = np.load(global_parameters_path, allow_pickle=True).item()
    #print("client{}起始参数的一个数：{}".format(client, list(global_parameters.values())[0][0][0][0]))
    traindata0 = np.load("data/train_data_part_{}.npy".format(client))
    traindata0 = torch.Tensor(traindata0).unsqueeze(1)
    trainlabels0 = np.load("data/train_labels_part_{}.npy".format(client))
    trainset0 = torch.utils.data.TensorDataset(torch.Tensor(traindata0), torch.Tensor(trainlabels0))
    trainloader = torch.utils.data.DataLoader(trainset0, batch_size=20, shuffle=True)
    # Load the test set
    testdata0 = np.load("data/test_data_part_{}.npy".format(client))
    testdata0 = torch.Tensor(testdata0).unsqueeze(1)
    testlabels0 = np.load("data/test_labels_part_{}.npy".format(client))
    testset0 = torch.utils.data.TensorDataset(torch.Tensor(testdata0), torch.Tensor(testlabels0))
    testloader = torch.utils.data.DataLoader(testset0, batch_size=20, shuffle=True)
    # 设备
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 模型
    net = LeNet()
    opti = torch.optim.SGD(net.parameters(), lr=0.01)  # lr学习率
    # local_epoch = 10  # 训练轮次
    local_batchsize = 20  # 训练批次
    # 调用客户端训练
    training_pro = ClientTrainer(client, trainloader, testloader, dev, net, opti, local_epoch, local_batchsize, )
    new_parameter = training_pro.train(global_parameters)  # 训练出新参数
    acc = training_pro.evaluate(new_parameter)
    print("client{} accuracy = {}".format(client, acc))
    return new_parameter, acc


def all_evaluate(new_parameter):
    testdata0 = np.load("data/Completed_dataset/test_data.npy")
    testdata0 = torch.Tensor(testdata0).unsqueeze(1)
    testlabels0 = np.load("data/Completed_dataset/test_labels.npy")
    testset0 = torch.utils.data.TensorDataset(torch.Tensor(testdata0), torch.Tensor(testlabels0))
    testloader = torch.utils.data.DataLoader(testset0, batch_size=20, shuffle=True)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    all_test = ClientTrainer(None, None, testloader, dev, net, None, 0, 20, )
    acc = all_test.evaluate(new_parameter)
    return acc


def net_to_vector(net_state_dict):
    flat_tensors = torch.cat([t.flatten() for t in net_state_dict.values()])
    net_shape = {name: list(s.shape) for name, s in net_state_dict.items()}
    return flat_tensors, net_shape


def vector_to_net(vec, net_shape):
    flag = 0
    net = {}
    for name, shape in net_shape.items():
        length = reduce(operator.mul, shape, 1)
        net[name] = torch.tensor(vec[flag:flag + length]).view(shape)
        flag += length
    return net


if __name__ == "__main__":
    client_number = 10
    # 生成初始参数
    net = LeNet()
    initial_parameters = {}
    if not initial_parameters:
        for key, var in net.state_dict().items():
            initial_parameters[key] = var.clone()
    np.save("initial_parameters.npy", initial_parameters)
    print("全局变量初始化完成")
    process_args = [{"global parameter path": "initial_parameters.npy", "client": cid, "local epoch": 5} for cid in
                    range(client_number)]
    # 定义聚合协议
    Pro = SecAgg(8)

    # 开始联邦学习
    agg_number = 10  # 聚合次数
    k = 0
    while k < agg_number:
        with Pool(4) as p:
            result = p.map(ClientTraining, process_args)
        print("准确率平均值：", sum([result[i][1] for i in range(client_number)]) / client_number)
        # 变形聚合
        shape = {}
        for i in result:
            vec, shape = net_to_vector(i[0])
            print(vec[0])
            Pro.add(vec)
        Pro.summary()
        # 聚合求平均
        aver = Pro.average()
        #print("聚合后全局参数的第一个数：", aver[0])
        global_parameters = vector_to_net(aver, shape)
        # 保存聚合后的参数
        np.save("global_parameters.npy", global_parameters)
        #print("聚合后全局参数的第一个数：{}".format(list(global_parameters.values())[0][0][0][0]))

        # 计算聚合后的模型在所有测试集下的准确率
        acc = all_evaluate(global_parameters)
        print("聚合后的模型在所有测试集下的准确率{:.8f}".format(acc))

        # 更新参数
        process_args = [{"global parameter path": "global_parameters.npy", "client": cid, "local epoch": 1} for cid in
                        range(client_number)]

        k += 1

    # 全部的测试集

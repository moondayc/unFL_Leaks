import numpy as np
import torch

from Model import LeNet
from Traning import ClientTrainer
from multiprocessing import Pool


def ClientTraining(client):
    # 读取训练和测试数据集
    global_parameters = np.load("../1/FederatedLearning/initial_parameters.npy", allow_pickle=True).item()
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
    local_epoch = 10  # 训练轮次
    local_batchsize = 20  # 训练批次
    # 调用客户端训练
    training_pro = ClientTrainer(client, trainloader, testloader, dev, net, opti, local_epoch, local_batchsize, )
    new_parameter = training_pro.train(global_parameters)  # 训练出新参数
    acc = training_pro.evaluate(new_parameter)
    print("client{} accuracy = {}".format(client, acc))
    return new_parameter, acc


if __name__ == "__main__":
    client_number = 2
    initial_parameters = {}
    net = LeNet()
    if not initial_parameters:
        for key, var in net.state_dict().items():
            initial_parameters[key] = var.clone()
    np.save("../1/FederatedLearning/initial_parameters.npy", initial_parameters)
    print("全局变量初始化完成")
    process_args = [(cid) for cid in range(client_number)]
    # print(global_parameters)
    print(process_args)
    with Pool(2) as p:
        result = p.map(ClientTraining, process_args)
    print(result[0][1])

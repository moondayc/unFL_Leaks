import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Model import LeNet


class ClientTrainer:
    def __init__(self, cid, trainloader, testloader, dev, net, opti, local_epoch=10, local_batchsize=128, loss_fun=nn.CrossEntropyLoss()):
        """
        :param trainloader: 本地训练数据集
        :param testloader: 本地测试数据集
        :param dev: 设备
        :param net: 网络结构
        :param opti: 优化器
        :param local_epoch: 本地训练轮次
        :param local_batchsize: 本地训练批次
        :param loss_fun: 损失函数
        """
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.dev = dev
        self.net = net.to(dev)
        self.opti = opti
        self.local_epoch = local_epoch
        self.local_batchsize = local_batchsize
        self.loss_fun = loss_fun

    def train(self, global_parameters):
        """
        :param global_parameters: 全局模型参数
        :return: 本地模型参数
        """
        self.net.load_state_dict(global_parameters, strict=True)
        self.net.train()
        for epoch in range(self.local_epoch):
            for data, label in tqdm(self.trainloader, desc=f'Client{self.cid} training epoch {epoch+1}/{self.local_epoch}'):
                data, label = data.to(self.dev), label.to(self.dev)
                self.opti.zero_grad()
                preds = self.net(data)
                loss = self.loss_fun(preds, label.long())
                loss.backward()
                self.opti.step()
        return self.net.state_dict()

    def evaluate(self, global_parameters):
        """
        :param global_parameters: 全局模型参数
        :return: 当前模型在测试集上的准确率
        """
        self.net.load_state_dict(global_parameters, strict=True)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, label in self.testloader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                _, predicted = torch.max(preds.data, 1)
                test_total += label.size(0)
                test_correct += (predicted == label).sum().item()
        test_acc = 100.0 * test_correct / test_total
        return test_acc

if __name__ == "__main__":
    client = 0
    global_parameters = {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Load the training set
    #trainset = torchvision.datasets.MNIST(root='data/MNIST', train=True, download=False, transform=transform)
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
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = LeNet()
    if not global_parameters:
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
            # global_parameters[key].requires_grad = True
    print("全局变量初始化完成")
    #print(global_parameters)
    opti = torch.optim.SGD(net.parameters(), lr=0.01)  # lr学习率
    local_epoch = 100
    local_batchsize = 20
    # lossfun = torch.nn.MSELoss()
    # trainloader, testloader, dev, net, opti, local_epoch = 10, local_batchsize = 128, loss_fun = nn.CrossEntropyLoss()
    for i in range(5):
        training_pro = ClientTrainer(i, trainloader, testloader, dev, net, opti, local_epoch, local_batchsize,)
        new_parameter = training_pro.train(global_parameters)  # 训练出新参数
        acc = training_pro.evaluate(new_parameter)
        print(acc)
        global_parameters = new_parameter



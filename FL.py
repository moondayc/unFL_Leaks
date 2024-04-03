import os

import numpy as np
import torch
from functools import reduce
import operator

from tqdm import tqdm

from Model import LeNet
from Traning import ClientTrainer
from multiprocessing import Pool
from SecAgg import SecAgg
import matplotlib.pyplot as plt

from utils import determining_original_model


def ClientTraining0(args):
    # 读取参数
    client = args["client"]
    local_epoch = args["local_epoch"]
    local_batch_size = args["local_batch_size"]
    # 读取全局参数和初始模型
    global_parameters_path = args["global_parameter_path"]
    original_model_name = args["original_model_name"]
    dataset_name = args["dataset_name"]
    # 训练数据集和测试集
    train_data_name = args["train_data"]
    train_label_name = args["train_label"]
    test_data_name = args["test_data"]
    test_label_name = args["test_label"]
    global_parameters = np.load(global_parameters_path, allow_pickle=True).item()
    # print("client{}起始参数的一个数：{}".format(client, list(global_parameters.values())[0][0][0][0]))
    # traindata0 = np.load("data/train_data_part_{}.npy".format(client))  # TODO：训练数据路径
    traindata0 = np.load(train_data_name.format(client))
    # traindata0 = torch.Tensor(traindata0).unsqueeze(1)
    # trainlabels0 = np.load("data/train_labels_part_{}.npy".format(client))  # TODO：训练数据标签路径
    trainlabels0 = np.load(train_label_name.format(client))
    trainset0 = torch.utils.data.TensorDataset(torch.Tensor(traindata0), torch.Tensor(trainlabels0))
    trainloader = torch.utils.data.DataLoader(trainset0, batch_size=local_batch_size, shuffle=True)
    # Load the test set
    testdata0 = np.load(test_data_name.format(client))
    # testdata0 = torch.Tensor(testdata0).unsqueeze(1)
    testlabels0 = np.load(test_label_name.format(client))
    testset0 = torch.utils.data.TensorDataset(torch.Tensor(testdata0), torch.Tensor(testlabels0))
    testloader = torch.utils.data.DataLoader(testset0, batch_size=local_batch_size, shuffle=True)
    # 设备
    dev = torch.device("cpu")
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 模型
    net = determining_original_model(original_model_name, dataset_name)
    # net.load_state_dict(global_parameters) 在调用train函数
    # opti = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.01, )  # lr学习率 lenet的
    # opti = torch.optim.SGD(net.parameters(), lr=0.01)
    opti = torch.optim.Adam(net.parameters(), lr=0.005)  # lr学习率 simplrCNN的
    # 调用客户端训练
    training_pro = ClientTrainer(client, trainloader, testloader, dev, net, opti, local_epoch, local_batch_size)
    # print("{} 训练之前第一个参数的值：{}".format(client, global_parameters["conv1.weight"][0][0][0]))
    new_parameter = training_pro.train(global_parameters)  # 训练出新参数
    # print("{} 训练之后第一个参数的值：{}".format(client, new_parameter["conv1.weight"][0][0][0]))
    acc = training_pro.evaluate(new_parameter)
    # print("client{} accuracy = {}".format(client, acc))
    return new_parameter, acc


def ClientTraining1(args):
    # 读取参数
    client = args["client"]
    local_epoch = args["local_epoch"]
    # 读取全局参数和初始模型
    global_parameters_path = args["global_parameter_path"]
    original_model_name = args["original_model_name"]
    dataset_name = args["dataset_name"]
    # 训练数据集和测试集
    train_data_name = args["train_data"]
    train_label_name = args["train_label"]
    test_data_name = args["test_data"]
    test_label_name = args["test_label"]
    global_parameters = np.load(global_parameters_path, allow_pickle=True).item()
    model = determining_original_model(original_model_name, dataset_name)
    model.load_state_dict(global_parameters)
    # print("{} 训练之前".format(client))
    # print(len(model.state_dict()["coef"]), len(model.state_dict()["coef"][0]))
    x = np.load(train_data_name.format(client))
    y = np.load(train_label_name.format(client))
    for i in range(local_epoch):
        model.train_model(x, y)
    new_parameter = model.state_dict()
    test_x = np.load(test_data_name.format(client))
    test_y = np.load(test_label_name.format(client))
    acc = model.test_model_acc(test_x, test_y)
    return new_parameter, acc


def all_evaluate0(new_parameter, model, test_path, participants, uid):
    if not uid:
        uid = []
    model.load_state_dict(new_parameter)
    testdata0 = [np.load(test_path[0].format(cid)) for cid in participants if cid not in uid]
    testdata = torch.Tensor(np.concatenate(testdata0))
    # testdata = testdata.unsqueeze(1)
    testlabels0 = [np.load(test_path[1].format(cid)) for cid in participants if cid not in uid]
    testlabels = torch.Tensor(np.concatenate(testlabels0))
    testset0 = torch.utils.data.TensorDataset(torch.Tensor(testdata), torch.Tensor(testlabels))
    testloader = torch.utils.data.DataLoader(testset0, batch_size=20, shuffle=False)
    test_correct = 0
    test_total = 0
    dev = torch.device("cpu")
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(dev), label.to(dev)
            preds = model(data)
            _, predicted = torch.max(preds.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()
    torch.cuda.empty_cache()
    acc = 100.0 * test_correct / test_total
    return acc


def all_evaluate1(new_parameter, model, test_path, participants, uid):
    if not uid:
        uid = []
    model.load_state_dict(new_parameter)
    testdata = [np.load(test_path[0].format(cid)) for cid in participants if cid not in uid]
    testdata = np.concatenate(testdata)
    testlabels = [np.load(test_path[1].format(cid)) for cid in participants if cid not in uid]
    testlabels = np.concatenate(testlabels)
    acc = model.test_model_acc(testdata, testlabels)
    return acc


# cid, trainloader, testloader, dev, net, opti, local_epoch=10, local_batchsize=128,
#                  loss_fun=nn.CrossEntropyLoss()


def net_to_vector(net_state_dict, model_name):
    if model_name in ["lenet", 'simpleCNN']:
        flat_tensors = torch.cat([t.flatten() for t in net_state_dict.values()])
        net_shape = {name: list(s.shape) for name, s in net_state_dict.items()}
    elif model_name in ["LR", "LR_without"]:
        flat_tensors = np.concatenate([t.flatten() for t in net_state_dict.values()])
        net_shape = {name: list(s.shape) for name, s in net_state_dict.items()}
    else:
        raise "model_name 无效"
    return flat_tensors, net_shape


def vector_to_net(vec, net_shape, model_name):
    if model_name in ["lenet", 'simpleCNN']:
        flag = 0
        net = {}
        for name, shape in net_shape.items():
            length = reduce(operator.mul, shape, 1)
            net[name] = torch.tensor(vec[flag:flag + length]).view(shape)
            flag += length
    elif model_name in ["LR", "LR_without"]:
        flag = 0
        net = {}
        for name, shape in net_shape.items():
            length = reduce(operator.mul, shape, 1)
            net[name] = np.array(vec[flag:flag + length]).reshape(shape)
            flag += length
    else:
        raise "model_name 无效"
    return net


class FederatedTrainer:
    def __init__(self, participants, original_model_name, dataset_name, initial_parameters_path, decimal_place, flag):
        self.participants = participants
        self.original_model_name = original_model_name
        self.dataset_name = dataset_name
        self.model = determining_original_model(self.original_model_name, self.dataset_name)
        self.initial_parameters_path = initial_parameters_path
        self.flag = flag
        self.decimal_place = decimal_place

    def training(self, max_agg_number, model_path, local_epoch, local_batch_size, data_path, unlearning_id=None):
        # model_path，训练的模型保存路径
        train_data, train_label, test_data, test_label, all_test = data_path
        if not unlearning_id:
            unlearning_id = []
        process_args = [
            {"global_parameter_path": self.initial_parameters_path, "client": cid, "local_epoch": local_epoch,
             "local_batch_size": local_batch_size, "original_model_name": self.original_model_name,
             "train_data": train_data.format(cid), "train_label": train_label.format(cid),
             "test_data": test_data.format(cid), "test_label": test_label.format(cid),
             "dataset_name": self.dataset_name
             }
            for cid in self.participants if cid not in unlearning_id]

        # 开始联邦学习
        # agg_number = 10  # 聚合次数
        k = 0
        acc_line = []
        participant_number = len(self.participants) - len(unlearning_id)
        # while k < max_agg_number:
        for k in tqdm(range(max_agg_number), desc="Federated Learning"):
            # TODO:LR部分修改
            if self.original_model_name in ["lenet", 'simpleCNN']:
                with Pool(3) as p:
                    result = p.map(ClientTraining0, process_args)
            elif self.original_model_name in ["LR", "LR_without"]:
                with Pool(4) as p:
                    result = p.map(ClientTraining1, process_args)
            else:
                raise "在联邦学习训练中没有这个模型"
            # aver_acc = sum([result[i][1] for i in range(participant_number)]) / participant_number
            # print("第{}次聚合准确率平均值：{}".format(k + 1, aver_acc))
            # 定义聚合协议， 精确到小数点后8位的聚合协议
            Pro = SecAgg(8)
            # 变形聚合
            shape = {}
            j = []
            for i in result:
                vec, shape = net_to_vector(i[0], self.original_model_name)
                j.append(vec[0])
                Pro.add(vec)
            Pro.summary()
            # 聚合求平均
            aver = Pro.average()
            # print("聚合后全局参数的第一个数：aver={}, 验算={}".format(aver[0], sum(j)/len(j)))
            global_parameters = vector_to_net(aver, shape, self.original_model_name)
            # 保存聚合后的参数
            np.save(model_path, global_parameters)
            # print("聚合后全局参数的第一个数：{}".format(list(global_parameters.values())[0][0][0][0]))

            # 计算聚合后的模型在所有测试集下的准确率
            if self.original_model_name in ['lenet', 'simpleCNN']:
                acc = all_evaluate0(global_parameters, self.model, all_test, self.participants, unlearning_id)
            elif self.original_model_name in ['LR', "LR_without"]:
                acc = all_evaluate1(global_parameters, self.model, all_test, self.participants, unlearning_id)
            else:
                raise "初始模型类型错误"
            acc_line.append(acc)
            print("第{}轮平均后的准确率: {}".format(k, acc))
            # 确定最新的参数
            # if len(acc_line) >= 2 and acc_line[-2] > 80 and abs(acc_line[-2] - acc_line[-1]) <= 10 ** (-self.decimal_place):
            if len(acc_line) >= 2 and acc_line[-2] > 80 and acc_line[-2] > acc_line[-1]:
                # TODO:这个条件怎么设置合理：1. 差距要小于10 ** (-self.decimal_place) 但是准确率出现了下降的现象
                #  2.找到最高点，出现下降就结束, 可是会出现先下降再上升的现象
                return k, acc_line[-2]
            # 更新参数
            for i in range(len(process_args)):
                process_args[i]["global_parameter_path"] = model_path
        return k, acc_line[-1]


if __name__ == "__main__":
    # pass
    dataset_name = "mnist"
    shadow_train_data_paths = "data/slice/{}/shadow_train_data_{{}}.npy".format(dataset_name)
    shadow_train_label_paths = "data/slice/{}/shadow_train_label_{{}}.npy".format(dataset_name)
    shadow_test_data_paths = "data/slice/{}/shadow_test_data_{{}}.npy".format(dataset_name)
    shadow_test_label_paths = "data/slice/{}/shadow_test_label_{{}}.npy".format(dataset_name)
    shadow_all_test_path = ["data/slice/{}/shadow_test_data_{{}}.npy".format(dataset_name),
                            "data/slice/{}/shadow_test_label_{{}}.npy".format(dataset_name)]
    client_number = [4, 17, 6, 3, 0, 2, 5, 9, 14]
    original_model_name = 'simpleCNN'
    agg_number = 2
    i = 1
    initial_parameters = {}

    shadow_initial_model_path0 = "shadow_initial_parameters.npy"
    shadow_original_model_path0 = "shadow_original_model.npy"
    initial_path = "model/{}_{}/".format(dataset_name, original_model_name)
    initial_shadow_path = initial_path + "shadow_models/"
    shadow_path = initial_shadow_path + str(i) + "/"
    shadow_initial_model_path = shadow_path + shadow_initial_model_path0
    shadow_original_model_path = shadow_path + shadow_original_model_path0
    local_epoch = 10
    local_batch_size = 100
    model = determining_original_model(original_model_name, dataset_name)
    if not initial_parameters:
        for key, var in model.state_dict().items():
            initial_parameters[key] = var.clone()
    # print(shadow_initial_model_path)
    np.save(shadow_initial_model_path, initial_parameters)

    data_path = [shadow_train_data_paths, shadow_train_label_paths, shadow_test_data_paths,
                 shadow_test_label_paths, shadow_all_test_path]
    ftrainer = FederatedTrainer(client_number, original_model_name, dataset_name,
                                shadow_initial_model_path, 8,
                                "shadow")
    k, acc = ftrainer.training(agg_number, shadow_original_model_path, local_epoch,
                               local_batch_size,
                               data_path)

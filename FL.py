import numpy as np
import torch
from functools import reduce
import operator
from Model import LeNet
from Traning import ClientTrainer
from multiprocessing import Pool
from SecAgg import SecAgg
import matplotlib.pyplot as plt

from utils import determining_original_model


def ClientTraining(args):
    # 读取参数
    client = args["client"]
    local_epoch = args["local_epoch"]
    local_batch_size = args["local_batch_size"]
    # 读取全局参数和初始模型
    global_parameters_path = args["global_parameter_path"]
    original_model_name = args["original_model_name"]
    # 训练数据集和测试集
    train_data_name = args["train_data"]
    train_label_name = args["train_label"]
    test_data_name = args["test_data"]
    test_label_name = args["test_label"]
    global_parameters = np.load(global_parameters_path, allow_pickle=True).item()
    # print("client{}起始参数的一个数：{}".format(client, list(global_parameters.values())[0][0][0][0]))
    # traindata0 = np.load("data/train_data_part_{}.npy".format(client))  # TODO：训练数据路径
    traindata0 = np.load(train_data_name.format(client))
    traindata0 = torch.Tensor(traindata0).unsqueeze(1)
    # trainlabels0 = np.load("data/train_labels_part_{}.npy".format(client))  # TODO：训练数据标签路径
    trainlabels0 = np.load(train_label_name.format(client))
    trainset0 = torch.utils.data.TensorDataset(torch.Tensor(traindata0), torch.Tensor(trainlabels0))
    trainloader = torch.utils.data.DataLoader(trainset0, batch_size=local_batch_size, shuffle=True)
    # Load the test set
    testdata0 = np.load(test_data_name.format(client))
    testdata0 = torch.Tensor(testdata0).unsqueeze(1)
    testlabels0 = np.load(test_label_name.format(client))
    testset0 = torch.utils.data.TensorDataset(torch.Tensor(testdata0), torch.Tensor(testlabels0))
    testloader = torch.utils.data.DataLoader(testset0, batch_size=local_batch_size, shuffle=True)
    # 设备
    dev = torch.device("cpu")
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 模型
    net = determining_original_model(original_model_name)
    net.load_state_dict(global_parameters)
    opti = torch.optim.SGD(net.parameters(), lr=0.01)  # lr学习率
    # 调用客户端训练
    training_pro = ClientTrainer(client, trainloader, testloader, dev, net, opti, local_epoch, local_batch_size, )
    new_parameter = training_pro.train(global_parameters)  # 训练出新参数
    acc = training_pro.evaluate(new_parameter)
    print("client{} accuracy = {}".format(client, acc))
    return new_parameter, acc


def all_evaluate(new_parameter, model, all_test_path):
    model.load_state_dict(new_parameter)
    testdata0 = np.load(all_test_path[0])
    testdata0 = torch.Tensor(testdata0).unsqueeze(1)
    testlabels0 = np.load(all_test_path[1])
    testset0 = torch.utils.data.TensorDataset(torch.Tensor(testdata0), torch.Tensor(testlabels0))
    testloader = torch.utils.data.DataLoader(testset0, batch_size=20, shuffle=True)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    all_test = ClientTrainer(None, None, testloader, dev, model, None, 0, 20, )
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


class FederatedTrainer:
    def __init__(self, client_number, original_model_name, initial_parameters_path):
        self.client_number = client_number
        self.original_model_name = original_model_name
        self.model = determining_original_model(self.original_model_name)
        self.initial_parameters_path = initial_parameters_path

    def training(self, max_agg_number, model_path, local_epoch, local_batch_size, data_path, unlearning_id=None):
        # model_path，训练的模型保存路径
        train_data, train_label, test_data, test_label, all_test = data_path
        if unlearning_id:
            process_args = [
                {"global_parameter_path": self.initial_parameters_path, "client": cid, "local_epoch": local_epoch,
                 "local_batch_size": local_batch_size, "original_model_name": self.original_model_name,
                 "train_data": train_data.format(cid), "train_label": train_label.format(cid),
                 "test_data": test_data.format(cid), "test_label": test_label.format(cid)
                 }
                for
                cid in range(self.client_number) if cid not in unlearning_id]
        else:
            process_args = [
                {"global_parameter_path": self.initial_parameters_path, "client": cid, "local_epoch": local_epoch,
                 "local_batch_size": local_batch_size, "original_model_name": self.original_model_name,
                 "train_data": train_data.format(cid), "train_label": train_label.format(cid),
                 "test_data": test_data.format(cid), "test_label": test_label.format(cid)} for
                cid
                in range(self.client_number)]
        # 定义聚合协议
        Pro = SecAgg(8)

        # 开始联邦学习
        # agg_number = 10  # 聚合次数
        k = 0
        acc_line = []
        participant_number = self.client_number - len(unlearning_id) if unlearning_id else self.client_number
        while k < max_agg_number:
            with Pool(4) as p:
                result = p.map(ClientTraining, process_args)
            aver_acc = sum([result[i][1] for i in range(participant_number)]) / participant_number
            print("第{}次聚合准确率平均值：{}".format(k + 1, aver_acc))
            # 变形聚合
            shape = {}
            for i in result:
                vec, shape = net_to_vector(i[0])
                Pro.add(vec)
            Pro.summary()
            # 聚合求平均
            aver = Pro.average()
            # print("聚合后全局参数的第一个数：", aver[0])
            global_parameters = vector_to_net(aver, shape)
            # 保存聚合后的参数
            np.save(model_path, global_parameters)
            # print("聚合后全局参数的第一个数：{}".format(list(global_parameters.values())[0][0][0][0]))

            # 计算聚合后的模型在所有测试集下的准确率

            acc = all_evaluate(global_parameters, self.model, all_test)
            if unlearning_id:
                print("unlearning id = {}".format(unlearning_id))
            print("完成第{}次聚合训练".format(k + 1))
            print("聚合后的模型在所有测试集下的准确率:{:.8f}".format(acc))
            acc_line.append(acc)
            # 确定最新的参数
            if len(acc_line) >= 2 and acc_line[-2] >= acc_line[-1]:
                print("训练结束，拟合成功，保留第{}轮聚合的参数，全局参数保存在{}".format(k, model_path))
                return k, acc_line[-2]
            # 更新参数
            for i in range(len(process_args)):
                process_args[i]["global parameter path"] = model_path

            k += 1
        print("训练结束，保留第{}轮聚合的参数，全局参数保存在{}".format(k, model_path))
        return k, acc_line[-1]
        # plt.plot(range(len(acc_line)), acc_line)
        # plt.show()


if __name__ == "__main__":
    client_number = 10
    net = LeNet()
    # initial_parameters = {}
    # if not initial_parameters:
    #     for key, var in net.state_dict().items():
    #         initial_parameters[key] = var.clone()
    # np.save("initial_parameters.npy", initial_parameters)
    print("全局变量初始化完成")
    FTrainer = FederatedTrainer(client_number, net, "model/shadow_initial_parameters.npy")
    agg_number = 10
    FTrainer.training(agg_number, "model/original_model.npy")

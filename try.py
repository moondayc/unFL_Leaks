import logging

import torch
import numpy as np

from argument import get_args
from exp import Exp
from Model import LeNet
from utils import determine_attack_model, determining_original_model


class A(Exp):
    def __init__(self, args):
        super(A, self).__init__(args)
        self.current_time = "16_23_32"
        self.model = determining_original_model(self.original_model_name)
        # shadow数据路径
        self.shadow_train_data_paths = "data/slice/shadow_train_data_{}.npy"
        self.shadow_train_label_paths = "data/slice/shadow_train_label_{}.npy"
        self.shadow_test_data_paths = "data/slice/shadow_test_data_{}.npy"
        self.shadow_test_label_paths = "data/slice/shadow_test_label_{}.npy"
        self.shadow_negative_data_path = "data/slice/shadow_negative_data.npy"
        self.shadow_all_test_path = ["data/slice/shadow_test_data_{}.npy",
                                     "data/slice/shadow_test_label_{}.npy"]
        # target数据路径
        self.target_train_data_paths = "data/slice/target_train_data_{}.npy"
        self.target_train_label_paths = "data/slice/target_train_label_{}.npy"
        self.target_test_data_paths = "data/slice/target_test_data_{}.npy"
        self.target_test_label_paths = "data/slice/target_test_label_{}.npy"
        self.target_negative_data_path = "data/slice/target_negative_data.npy"
        self.target_all_test_path = ["data/slice/target_test_data_{}.npy",
                                     "data/slice/target_test_label_{}.npy"]
        # 模型路径
        self.shadow_initial_model_path = "model/{}/shadow_initial_parameters.npy".format(self.current_time)
        self.shadow_original_model_path = "model/{}/shadow_original_model.npy".format(self.current_time)
        self.shadow_unlearning_model_path = "model/{}/shadow_unlearned_model_unid_{{}}.npy".format(self.current_time)

        self.target_initial_model_path = "model/{}/target_initial_parameters.npy".format(self.current_time)
        self.target_original_model_path = "model/{}/target_original_model.npy".format(self.current_time)
        self.target_unlearning_model_path = "model/{}/target_unlearned_model_unid_{{}}.npy".format(self.current_time)

        self.attack_model_path = "model/{}/{}_attacker.npy".format(self.current_time, self.attack_model_name)
        self.model = determining_original_model(self.original_model_name)
        self.attack_model = determine_attack_model(self.attack_model_name)
        x, y = self.construct_dataset(0)
        self.training_attack_model(x, y)
        x, y = self.construct_dataset(1)  # 构造测试攻击模型的数据集

        self.attack_model.load_model(self.attack_model_path)
        self.evaluate_attack_model(x, y)

    def training_attack_model(self, x, y):
        self.logger.info("开始训练攻击模型......")
        self.attack_model = determine_attack_model(self.attack_model_name)
        attack_model_path = self.attack_model_path.format(self.attack_model_name)
        self.attack_model.train_model(x, y, attack_model_path)
        self.logger.info("{} 攻击模型训练完成！模型保存在 {} ".format(self.attack_model_name, attack_model_path))


    def evaluate_attack_model(self, x, y):
        print(1)
        acc = self.attack_model.test_model_acc(x, y)
        self.logger.info("{} 攻击模型的准确率 acc={}".format(self.attack_model_name, acc))
        auc = self.attack_model.test_model_auc(x, y)
        self.logger.info("{} 模型 auc = {}".format(self.attack_model_name, auc))

    def construct_dataset(self, flag):
        # 构造数据集
        self.logger.info("开始{}攻击模型数据集......".format("构造" if not flag else "测试"))
        features = []
        positive_n = 0
        # 构造正向数据集
        for uid in range(self.unlearning_round):
            feature = self.construct_positive_data(uid, flag)
            # print("feature")
            # print(feature)
            positive_n += len(feature)
            features.append(feature)
        positive_labels = [1] * positive_n
        negative_n = 0
        # 构造反向数据集
        for uid in range(self.unlearning_round):
            feature = self.construct_negative_data(uid, flag)
            negative_n += len(feature)
            features.append(feature)
        negative_labels = [0] * negative_n
        features = sum(features, [])
        labels = positive_labels + negative_labels
        print("正向数据量: {}  反向数据量: {}".format(positive_n, negative_n))
        self.logger.info("攻击模型{}完成".format("训练集" if not flag else "测试集"))
        return features, labels

    def construct_positive_data(self, uid, flag):
        # 构造uid客户端的的输出，即正向数据
        data_paths = self.shadow_train_data_paths.format(uid) if not flag else self.target_train_data_paths.format(uid)
        data = np.load(data_paths, allow_pickle=True)  # uid客户端的数据，正向数据
        data = torch.Tensor(data).unsqueeze(1)
        positive_feature = self.get_differential_feature(uid, data, flag)
        print("每一次的正向数据: {}".format(len(positive_feature)))
        return positive_feature

    def construct_negative_data(self, uid, flag):
        data_paths = self.shadow_negative_data_path if not flag else self.target_negative_data_path

        data = np.load(data_paths, allow_pickle=True)  # 反向数据
        data = torch.Tensor(data).unsqueeze(1)
        negative_feature = self.get_differential_feature(uid, data, flag)
        return negative_feature

    def get_model_data(self, model_path, data):
        # 得到data输入对应模型的结果
        model_parameter = np.load(model_path, allow_pickle=True).item()
        self.model.load_state_dict(model_parameter)
        preds = self.model(data)
        return preds

    def get_differential_feature(self, uid, data, flag):
        # 构造数据在初始模型和uid去学习模型输出的差异

        original_model = self.shadow_original_model_path if not flag else self.target_original_model_path
        unlearning_model_paths = self.shadow_unlearning_model_path.format(
            uid) if not flag else self.target_unlearning_model_path.format(uid)
        original_preds = self.get_model_data(original_model, data)
        unleared_preds = self.get_model_data(unlearning_model_paths, data)
        differential_data = original_preds - unleared_preds
        # dataset = torch.utils.data.TensorDataset(torch.Tensor(differential_data), torch.Tensor(label))
        return differential_data.tolist()


if __name__ == "__main__":
    args = get_args()
    a = A(args)

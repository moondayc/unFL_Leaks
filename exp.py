import logging
import os.path
import random
import pickle
import re

import numpy as np
import pandas as pd
import torch.utils.data
from tqdm import tqdm
from FL import FederatedTrainer
from Model import LeNet
from argument import get_args
from utils import determining_original_model, determine_attack_model
from datetime import datetime


class Exp:
    def __init__(self, args):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("trainer")
        self.dataset_name = args["dataset_name"]
        self.original_model_name = args["original_model_name"]
        self.unlearning_round = args["unlearning round"]
        self.max_agg_round = args["max_aggregation_round"]
        self.local_batch_size = args["local_batch_size"]
        self.local_epoch = args["local_epoch"]
        self.client_number = args["client_number"]
        self.decimal_place = args["decimal_places"]
        self.all_client_number = args["all_client_number"]
        self.attack_model_name = args["attack_model_name"]
        self.round_number = args["round_number"]
        assert self.unlearning_round <= self.client_number, "去学习个数应该小于客户端个数"
        self.current_time = datetime.now().strftime("%d_%H_%M")
        # self.current_time = "18_01_06"
        file_handler = logging.FileHandler(
            'log/{}_{}_{}_{}.log'.format(self.current_time, self.dataset_name, self.original_model_name,
                                         self.attack_model_name))
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.attack_model = None
        self.original_attack_model = None
        self.initial_path = "model/{}_{}/".format(self.dataset_name, self.original_model_name)
        if not os.path.exists(self.initial_path):
            os.makedirs(self.initial_path)
            self.logger.info("生成模型目录: {} ".format(self.initial_path))

        self.logger.info(args)
        self.logger.info("Experiment Start!".format())
        self.load_data()

    def load_data(self):
        self.logger.info('loading data')

        self.logger.info('loaded data')


class ModelTrainer(Exp):
    def __init__(self, args):
        super(ModelTrainer, self).__init__(args)
        self.model = determining_original_model(self.original_model_name)

        self.shadow_train_data_paths = "data/slice/shadow_train_data_{}.npy"
        self.shadow_train_label_paths = "data/slice/shadow_train_label_{}.npy"
        self.shadow_test_data_paths = "data/slice/shadow_test_data_{}.npy"
        self.shadow_test_label_paths = "data/slice/shadow_test_label_{}.npy"
        self.shadow_all_test_path = ["data/slice/shadow_test_data_{}.npy",
                                     "data/slice/shadow_test_label_{}.npy"]
        # target数据路径
        self.target_train_data_paths = "data/slice/target_train_data_{}.npy"
        self.target_train_label_paths = "data/slice/target_train_label_{}.npy"
        self.target_test_data_paths = "data/slice/target_test_data_{}.npy"
        self.target_test_label_paths = "data/slice/target_test_label_{}.npy"
        self.target_all_test_path = ["data/slice/target_test_data_{}.npy",
                                     "data/slice/target_test_label_{}.npy"]
        # 模型路径

        self.shadow_initial_model_path0 = "shadow_initial_parameters.npy"
        self.shadow_original_model_path0 = "shadow_original_model.npy"
        self.shadow_unlearning_model_path0 = "shadow_unlearned_model_unid_{}.npy"

        self.target_initial_model_path0 = "target_initial_parameters.npy"
        self.target_original_model_path0 = "target_original_model.npy"
        self.target_unlearning_model_path0 = "target_unlearned_model_unid_{}.npy"

        # 参与shadow客户端的id
        self.shadow_negative_id = {}
        self.shadow_participants_id = {}
        self.shadow_unlearned_ids = {}

        self.target_negative_id = {}
        self.target_participants_id = {}
        self.target_unlearned_ids = {}

        # shadow数据路径
        self.get_shadow_models(self.round_number)  # 训练多轮shadow模型
        self.get_target_models(self.round_number)  # 训练target模型

    def get_shadow_models(self, n):
        self.initial_shadow_path = self.initial_path + "shadow_models/"
        if not self.initial_shadow_path:
            os.makedirs(self.initial_shadow_path)
            self.logger.info("成功生成目录: {}".format(self.initial_shadow_path))
        for i in tqdm(range(n), desc="shadow training round"):
            self.shadow_path = self.initial_shadow_path + str(i) + "/"
            if not os.path.exists(self.shadow_path):
                os.makedirs(self.shadow_path)
            self.shadow_initial_model_path = self.shadow_path + self.shadow_initial_model_path0
            self.shadow_original_model_path = self.shadow_path + self.shadow_original_model_path0
            self.shadow_unlearning_model_path = self.shadow_path + self.shadow_unlearning_model_path0
            random_ids = random.sample(range(20), 11)
            self.shadow_negative_id[i] = random_ids[-1]
            self.shadow_participants_id[i] = random_ids[:-1]
            self.logger.info("shadow: 第 {i} 轮 参与训练的客户端id = {ids} ".format(i=i, ids=self.shadow_participants_id[i]))
            self.logger.info("shadow: 第 {i} 轮 反向数据客户端id = {ids}".format(i=i, ids=self.shadow_negative_id[i]))
            self.training_shadow_model(i)
            with open(self.shadow_path + "negative_id.txt", "w") as f:
                f.write(str(self.shadow_negative_id[i]))
                f.write("\n")
                f.write(str(self.shadow_participants_id[i]))

    def get_target_models(self, n):
        self.initial_target_path = self.initial_path + "target_models/"
        if not self.initial_target_path:
            os.makedirs(self.initial_target_path)
            self.logger.info("成功生成目录: {}".format(self.initial_target_path))
        for i in tqdm(range(n), desc="target training round"):
            self.target_path = self.initial_target_path + str(i) + "/"
            if not os.path.exists(self.target_path):
                os.makedirs(self.target_path)
            print(self.target_path)
            self.target_initial_model_path = self.target_path + self.target_initial_model_path0
            self.target_original_model_path = self.target_path + self.target_original_model_path0
            self.target_unlearning_model_path = self.target_path + self.target_unlearning_model_path0
            random_ids = random.sample(range(20), 11)
            self.target_negative_id[i] = random_ids[-1]
            self.target_participants_id[i] = random_ids[:-1]
            # self.target_negative_data_path = self.target_negative_data_path.format(self.target_negative_id[i])
            self.logger.info("target: 第 {i} 轮 参与训练的客户端id = {ids} ".format(i=i, ids=self.target_participants_id[i]))
            self.logger.info("target: 第 {i} 轮 反向数据客户端id = {ids}".format(i=i, ids=self.target_negative_id[i]))
            self.training_target_model(i)
            with open(self.target_path + "negative_id.txt", "w") as f:
                f.write(str(self.shadow_negative_id[i]))
                f.write("\n")
                f.write(str(self.shadow_participants_id[i]))

    def training_shadow_model(self, n):  # 训练shadow模型
        initial_parameters = {}
        if not initial_parameters:
            for key, var in self.model.state_dict().items():
                initial_parameters[key] = var.clone()
        np.save(self.shadow_initial_model_path, initial_parameters)
        self.logger.info("shadow {}：全局变量初始化完成".format(n))
        ftrainer = FederatedTrainer(self.shadow_participants_id[n], self.original_model_name,
                                    self.shadow_initial_model_path, self.decimal_place,
                                    "shadow")
        data_path = [self.shadow_train_data_paths, self.shadow_train_label_paths, self.shadow_test_data_paths,
                     self.shadow_test_label_paths, self.shadow_all_test_path]
        k, acc = ftrainer.training(self.max_agg_round, self.shadow_original_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path)
        self.logger.info("shadow {}:初始模型聚合{}轮次, 全训练模型准确率为{}".format(n, k, acc))
        self.logger.info('shadow {}:初始模型训练完成'.format(n))

        self.logger.info('{}: 开始训练shadow去学习模型.....'.format(n))
        un = random.sample(self.shadow_participants_id[n], 1)[0]
        self.shadow_unlearned_ids[n] = un
        self.logger.info("shadow: unlearning id = {}".format(un))
        unlearning_model_path = self.shadow_unlearning_model_path.format(un)
        k, acc = ftrainer.training(self.max_agg_round, unlearning_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path, [un])
        self.logger.info("shadow {}: {}去学习模型聚合{}轮次, 全训练模型准确率为{}".format(n, un, k, acc))
        self.logger.info('shadow {}: 模型训练完成'.format(n))

    def training_target_model(self, j):
        initial_parameters = {}
        self.model.initialize_parameters()  # 模型参数随机化
        if not initial_parameters:
            for key, var in self.model.state_dict().items():  # 此时的self.model是经过训练的
                initial_parameters[key] = var.clone()
        np.save(self.target_initial_model_path, initial_parameters)
        self.logger.info("target 全局变量初始化完成")
        self.logger.info("开始训练target模型......")
        ttrainer = FederatedTrainer(self.target_participants_id[j], self.original_model_name,
                                    self.target_initial_model_path, self.decimal_place,
                                    "target")
        data_path = [self.target_train_data_paths, self.target_train_label_paths, self.target_test_data_paths,
                     self.target_test_label_paths, self.target_all_test_path]
        k, acc = ttrainer.training(self.max_agg_round, self.target_original_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path)
        self.logger.info("target:初始模型聚合{}轮次, 全训练模型准确率为{}".format(k, acc))
        self.logger.info('target:初始模型训练完成')

        self.logger.info('开始训练target去学习模型.....')
        un = random.sample(self.target_participants_id[j], 1)[0]
        self.target_unlearned_ids[j] = un
        self.logger.info("target: unlearning id = {}".format(un))
        unlearning_model_path = self.target_unlearning_model_path.format(un)
        k, acc = ttrainer.training(self.max_agg_round, unlearning_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path, [un])
        self.logger.info("target第{}个去学习模型聚合{}轮次, 全训练模型准确率为{}".format(un, k, acc))
        self.logger.info('target:模型训练完成')


class AttackModelTrainer(Exp):
    def __init__(self, args):
        super(AttackModelTrainer, self).__init__(args)
        self.shadow_train_data_paths = "data/slice/shadow_train_data_{}.npy"
        self.target_train_data_paths = "data/slice/target_train_data_{}.npy"

        self.original_model_path = "shadow_original_model.npy"
        self.original_attack_model_path = self.initial_path+"original_{}_attacker.npy"
        self.attack_model_path = self.initial_path + "{}_attacker.npy"

        # 不考虑数据删除场景
        base_x, base_y = self.construct_base_dataset(0, self.round_number, "shadow")  # 构造训练攻击模型的数据集
        self.original_attack_model = self.training_attack_model(base_x, base_y, 0)  # 训练攻击模型

        test_base_x, test_base_y = self.construct_base_dataset(0, self.round_number, "target")  # 构造攻击模型的测试集
        self.evaluate_attack_model(test_base_x, test_base_y, 0)
        # 考虑数据删除场景
        x, y = self.construct_diff_dataset(0, self.round_number, "shadow")  # 构造训练攻击模型的数据集
        self.attack_model = self.training_attack_model(x, y, 1)  # 训练攻击模型
        test__x, test_y = self.construct_diff_dataset(0, self.round_number, "target")
        self.evaluate_attack_model(test__x, test_y, 1)

        # 计算degcount, degrate
        degcount, degrate = self.calculate_DegCount_and_DegRate(test_base_x, test_base_y, self.original_attack_model,
                                                                test__x, test_y, self.attack_model)
        print("degcount = {} , degrate = {}".format(degcount, degrate))

    def construct_base_dataset(self, begin, end, flag):
        # flag :["shadow", "target"]
        features = []
        labels = []
        for j in range(begin, end):
            # 构造正向特征
            initial_path = self.initial_path + flag + "_models/" + str(j) + "/"
            model_path, _, unid, negative_id = self.get_path(initial_path)
            print("444444444")
            print(model_path, _, unid, negative_id )
            positive_data_path = self.shadow_train_data_paths.format(
                unid) if flag == "shadow" else self.target_train_data_paths.format(unid)
            print("11111111111", positive_data_path)
            feature = self.get_model_output(model_path, positive_data_path).tolist()
            features.append(feature)
            label = [1] * len(feature)
            labels.append(label)
            # 构造反向数据集
            negitive_data_path = self.shadow_train_data_paths.format(
                negative_id) if flag == "shadow" else self.target_train_data_paths.format(negative_id)
            feature = self.get_model_output(model_path, negitive_data_path).tolist()
            features.append(feature)
            label = [0] * len(feature)
            labels.append(label)
        features = sum(features, [])
        labels = sum(labels, [])
        return features, labels

    def construct_diff_dataset(self, begin, end, flag):
        # flag :["shadow", "target"]
        features = []
        labels = []
        for j in range(begin, end):
            # 构造正向特征
            initial_path = self.initial_path + flag + "_models/" + str(j) + "/"
            model_path, un_model_path, unid, negative_id = self.get_path(initial_path)
            positive_data_path = self.shadow_train_data_paths.format(
                unid) if flag == "shadow" else self.target_train_data_paths.format(unid)
            feature = self.get_differential_data(model_path, un_model_path, positive_data_path).tolist()
            features.append(feature)
            label = [1] * len(feature)
            labels.append(label)
            # 构造反向数据集
            negitive_data_path = self.shadow_train_data_paths.format(
                negative_id) if flag == "shadow" else self.target_train_data_paths.format(negative_id)
            feature = self.get_differential_data(model_path, un_model_path, positive_data_path).tolist()
            features.append(feature)
            label = [0] * len(feature)
            labels.append(label)
        features = sum(features, [])
        labels = sum(labels, [])
        return features, labels

    def get_path(self, initial_path):
        files_list = os.listdir(initial_path)
        original_model_path = [i for i in files_list if re.search("original_model.npy", i)][0]
        unlearned_model_path = [i for i in files_list if re.search("_unlearned_model_unid", i)][0]
        unid = unlearned_model_path[len("shadow_unlearned_model_unid_"):-4]
        with open(initial_path + '/' + "negative_id.txt") as f:
            negative_id = int(f.readline())
        return initial_path+original_model_path, initial_path+unlearned_model_path, unid, negative_id

    def get_differential_data(self, model_path, unlearned_path, data_path):
        original_output = self.get_model_output(model_path, data_path)
        unlearned_output = self.get_model_output(unlearned_path, data_path)
        diff_output = unlearned_output - original_output
        return diff_output

    def get_model_output(self, model_path, data_path):
        data = np.load(data_path, allow_pickle=True)
        model = determining_original_model(self.original_model_name)
        model_parameter = np.load(model_path, allow_pickle=True).item()
        model.load_state_dict(model_parameter)
        data = torch.Tensor(data).unsqueeze(1)
        feature = model(data)
        return feature

    def training_attack_model(self, x, y, flag):
        model_flag = "考虑删除数据删除场景" if flag else "不考虑数据删除场景"
        attack_model_path = self.attack_model_path if flag else self.original_attack_model_path
        self.logger.info("{}, 开始训练攻击模型......".format(model_flag))
        attack_model = determine_attack_model(self.attack_model_name)
        attack_model_path = attack_model_path.format(self.attack_model_name)
        attack_model.train_model(x, y, attack_model_path)
        self.logger.info("{}, {} 攻击模型训练完成！模型保存在 {} ".format(model_flag, self.attack_model_name, attack_model_path))
        return attack_model

    def evaluate_attack_model(self, x, y, flag):
        model = self.attack_model if flag else self.original_attack_model
        acc = model.test_model_acc(x, y)
        model_flag = "考虑删除数据删除场景" if flag else "不考虑数据删除场景"
        self.logger.info("{}, {} 攻击模型的准确率 acc={}".format(model_flag, self.attack_model_name, acc))
        auc = model.test_model_auc(x, y)
        self.logger.info("{}, {} 模型 auc = {}".format(model_flag, self.attack_model_name, auc))

    def calculate_DegCount_and_DegRate(self, test_base_x, test_base_y, base_model, test_x, test_y, model):
        assert test_base_y == test_y, "两个攻击模型的测试数据必须相同"
        test_base_x, test_base_y = np.array(test_base_x), np.array(test_base_y)
        test_x, test_y = np.array(test_x), np.array(test_y)

        base_model_pred = base_model.predict_proba(test_base_x)
        base_model_pred = pd.DataFrame(base_model_pred)

        model_pred = model.predict_proba(test_x)
        model_pred = pd.DataFrame(model_pred)

        diff = model_pred - base_model_pred
        diff_1 = (diff[test_y == 1][1] > 0).sum()
        diff_0 = (diff[test_y == 0][1] < 0).sum()
        degcount = (diff_1 + diff_0) / len(test_y)
        diff_1 = diff[test_y == 1][1].sum()
        diff_0 = -diff[test_y == 0][1].sum()
        degrate = (diff_1 + diff_0) / len(test_y)
        print(degcount, degrate)
        return degcount, degrate

if __name__ == "__main__":
    args = get_args()
    AttackModelTrainer(args)

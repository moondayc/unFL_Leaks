import logging
import os.path
import random
import pickle
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch.utils.data
from sklearn import preprocessing
from tqdm import tqdm
from FL import FederatedTrainer
from Model import LeNet
from argument import get_args
from utils import determining_original_model, determine_attack_model
from datetime import datetime


class Exp:
    def __init__(self, args):
        self.dataset_name = args["dataset_name"]
        self.original_model_name = args["original_model_name"]
        # self.unlearning_round = args["unlearning round"]
        self.max_agg_round = args["max_aggregation_round"]
        self.local_batch_size = args["local_batch_size"]
        self.local_epoch = args["local_epoch"]
        self.client_number = args["client_number"]
        self.decimal_place = args["decimal_places"]
        self.all_client_number = args["all_client_number"]
        self.attack_model_name = args["attack_model_name"]
        self.round_number = args["round_number"]
        assert self.client_number <= self.all_client_number, "每一轮参与客户端训练的参数必须小于客户端总数"

        self.current_time = datetime.now().strftime("%d_%H_%M")
        # self.current_time = "18_01_06"

        self.attack_model = None
        self.original_attack_model = None
        self.initial_path = "model/{}_{}/".format(self.dataset_name, self.original_model_name)

        # TODO：便于控制中间模型模型训练
        self.begin, self.end = 0, 1

    def load_data(self):
        # 数据集已经下载到data/slice
        self.logger.info('loading data')
        self.logger.info('loaded data')


class ModelTrainer(Exp):
    def __init__(self, args):
        super(ModelTrainer, self).__init__(args)
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("trainer")
        if not os.path.exists(self.initial_path):
            os.makedirs(self.initial_path)
            self.logger.info("生成模型目录: {} ".format(self.initial_path))
        path = 'log/trainer_{}_{}/'.format(self.dataset_name, self.original_model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        file_handler = logging.FileHandler(path + '{}.log'.format(self.current_time))
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.info(args)
        self.logger.info("Experiment Start!".format())
        self.load_data()
        self.logger.info("begin = {}, end = {}".format(self.begin, self.end))

        self.shadow_train_data_paths = "data/slice/{}/shadow_train_data_{{}}.npy".format(self.dataset_name)
        self.shadow_train_label_paths = "data/slice/{}/shadow_train_label_{{}}.npy".format(self.dataset_name)
        self.shadow_test_data_paths = "data/slice/{}/shadow_test_data_{{}}.npy".format(self.dataset_name)
        self.shadow_test_label_paths = "data/slice/{}/shadow_test_label_{{}}.npy".format(self.dataset_name)
        self.shadow_all_test_path = ["data/slice/{}/shadow_test_data_{{}}.npy".format(self.dataset_name),
                                     "data/slice/{}/shadow_test_label_{{}}.npy".format(self.dataset_name)]
        # target数据路径
        self.target_train_data_paths = "data/slice/{}/target_train_data_{{}}.npy".format(self.dataset_name)
        self.target_train_label_paths = "data/slice/{}/target_train_label_{{}}.npy".format(self.dataset_name)
        self.target_test_data_paths = "data/slice/{}/target_test_data_{{}}.npy".format(self.dataset_name)
        self.target_test_label_paths = "data/slice/{}/target_test_label_{{}}.npy".format(self.dataset_name)
        self.target_all_test_path = ["data/slice/{}/target_test_data_{{}}.npy".format(self.dataset_name),
                                     "data/slice/{}/target_test_label_{{}}.npy".format(self.dataset_name)]
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
        self.model = determining_original_model(self.original_model_name, self.dataset_name)
        self.get_shadow_models(self.round_number)  # 训练多轮shadow模型
        self.get_target_models(self.round_number)  # 训练target模型

    def get_shadow_models(self, n):
        """
        :param n:  shadow 模型的个数
        :return: 训练出来的模型保存在
        self.shadow_initial_model_path， self.shadow_original_model_path，self.shadow_unlearning_model_path
        """
        self.initial_shadow_path = self.initial_path + "shadow_models/"
        if not os.path.exists(self.initial_shadow_path):
            os.makedirs(self.initial_shadow_path)
            self.logger.info("成功生成目录: {}".format(self.initial_shadow_path))
        for i in tqdm(range(n), desc="shadow training round"):
        # for i in tqdm(range(self.begin, self.end), desc="shadow training round"): #便于控制中间模型训练
            self.shadow_path = self.initial_shadow_path + str(i) + "/"
            if not os.path.exists(self.shadow_path):
                os.makedirs(self.shadow_path)
            else:
                self.clear_directory(self.shadow_path)
            self.logger.info("shadow模型文件保存在：{}".format(self.shadow_path))
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
                f.write("\n")
                f.write("uid = " + str(self.shadow_unlearned_ids[i]))

    def get_target_models(self, n):
        self.initial_target_path = self.initial_path + "target_models/"
        if not os.path.exists(self.initial_target_path):
            os.makedirs(self.initial_target_path)
            self.logger.info("成功生成目录: {}".format(self.initial_target_path))
        for i in tqdm(range(n), desc="target training round"):
        # for i in tqdm(range(self.begin, self.end), desc="target training round"): # 便于控制中间模型训练
            self.target_path = self.initial_target_path + str(i) + "/"
            if not os.path.exists(self.target_path):
                os.makedirs(self.target_path)
            else:
                self.clear_directory(self.target_path)
            self.logger.info("target模型文件保存在：{}".format(self.target_path))
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
                f.write(str(self.target_negative_id[i]))
                f.write("\n")
                f.write(str(self.target_participants_id[i]))
                f.write("\n")
                f.write("uid = " + str(self.target_unlearned_ids[i]))

    def clear_directory(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def training_shadow_model(self, n):  # 训练shadow模型
        initial_parameters = {}
        if self.original_model_name in ['lenet', 'simpleCNN']:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var.clone()
        elif self.original_model_name in ['LR', "LR_without"]:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var
        else:
            raise "exp 初始化中没有这个模型"
        np.save(self.shadow_initial_model_path, initial_parameters)
        self.logger.info("shadow {}：全局变量初始化完成".format(n))
        ftrainer = FederatedTrainer(self.shadow_participants_id[n], self.original_model_name, self.dataset_name,
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
        if self.original_model_name in ['lenet', 'simpleCNN']:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var.clone()
        elif self.original_model_name in ['LR', "LR_without"]:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var
        else:
            raise "training_target_model 初始化没有该模型"
        np.save(self.target_initial_model_path, initial_parameters)
        self.logger.info("target 全局变量初始化完成")
        self.logger.info("开始训练target模型......")
        ttrainer = FederatedTrainer(self.target_participants_id[j], self.original_model_name, self.dataset_name,
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
        for self.end in range(1, 11):
            tmp_n = self.end - self.begin
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger("Attacker")
            if not os.path.exists(self.initial_path):
                os.makedirs(self.initial_path)
                self.logger.info("生成模型目录: {} ".format(self.initial_path))
            path = 'log/attacker_{}_{}_{}/'.format(self.dataset_name, self.original_model_name, self.attack_model_name)
            if not os.path.exists(path):
                os.makedirs(path)
            file_handler = logging.FileHandler(path + '{}_{}.log'.format(tmp_n, self.current_time))
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

            self.logger.info(args)
            self.logger.info("Experiment Start!".format())
            self.load_data()

            self.logger.info("初始模型个数: {}".format(tmp_n))  # TODO：tmp_n 要用于记录参与的训练初始模型数
            self.logger.info("begin = {}, end = {}".format(self.begin, self.end))

            self.shadow_train_data_paths = "data/slice/{}/shadow_train_data_{{}}.npy".format(self.dataset_name)
            self.target_train_data_paths = "data/slice/{}/target_train_data_{{}}.npy".format(self.dataset_name)

            self.original_model_path = "shadow_original_model.npy"
            attacker_path = self.initial_path + "{}/".format(self.attack_model_name)
            if not os.path.exists(attacker_path):
                os.makedirs(attacker_path)
            self.original_attack_model_path = attacker_path + "original_{{}}_{}_attacker.npy".format(tmp_n)
            self.attack_model_path = attacker_path + "{{}}_{}_attacker.npy".format(tmp_n)

            # 不考虑数据删除场景
            base_x, base_y = self.construct_base_dataset(self.begin, self.end, "shadow")  # 构造训练攻击模型的数据集
            self.original_attack_model = self.training_attack_model(base_x, base_y, 0)  # 训练攻击模型
            test_base_x, test_base_y = self.construct_base_dataset(self.begin, self.end, "target")  # 构造攻击模型的测试集
            self.evaluate_attack_model(test_base_x, test_base_y, 0)
            # data = np.concatenate([base_x, test_base_x])
            # scaler = preprocessing.StandardScaler().fit(data)
            # with open('data/{}/{}_{}_attacker_scaler.pkl'.format(self.dataset_name, "base", self.original_model_name),
            #           'wb') as file:
            #     pickle.dump(scaler, file)

            # 考虑数据删除场景
            x, y = self.construct_diff_dataset(self.begin, self.end, "shadow")  # 构造训练攻击模型的数据集
            self.attack_model = self.training_attack_model(x, y, 1)  # 训练攻击模型
            test_x, test_y = self.construct_diff_dataset(self.begin, self.end, "target")
            # self.evaluate_attack_model(x, y, 1)
            self.evaluate_attack_model(test_x, test_y, 1)
            # data = np.concatenate([x, test_x])
            # scaler = preprocessing.StandardScaler().fit(data)
            # with open('data/{}/{}_{}_attacker_scaler.pkl'.format(self.dataset_name, "new", self.original_model_name),
            #           'wb') as file:
            #     pickle.dump(scaler, file)

            # 计算degcount, degrate
            degcount, degrate = self.calculate_DegCount_and_DegRate(test_base_x, test_base_y, self.original_attack_model,
                                                                    test_x, test_y, self.attack_model)
            self.logger.info("degcount = {} , degrate = {}".format(degcount, degrate))

    def construct_base_dataset(self, begin, end, flag):
        # flag :["shadow", "target"]
        features = []
        labels = []
        for j in range(begin, end):
            # 构造正向特征
            initial_path = self.initial_path + flag + "_models/" + str(j) + "/"
            model_path, _, unid, negative_id = self.get_path(initial_path)
            positive_data_path = self.shadow_train_data_paths.format(
                unid) if flag == "shadow" else self.target_train_data_paths.format(unid)
            #print("model_path={}, data_path={}".format(model_path, positive_data_path))
            feature = self.get_model_output(model_path, positive_data_path).tolist()
            features.append(feature)
            label = [1] * len(feature)
            labels.append(label)
            # 构造反向数据集
            negative_data_path = self.shadow_train_data_paths.format(
                negative_id) if flag == "shadow" else self.target_train_data_paths.format(negative_id)
            feature = self.get_model_output(model_path, negative_data_path).tolist()
            # print("model_path={}, data_path={}".format(model_path, negative_data_path))
            features.append(feature)
            label = [0] * len(feature)
            labels.append(label)
        features = sum(features, [])
        labels = sum(labels, [])
        print("{}: 正向数据量: {}; 反向数据量: {}".format(flag, sum(labels), len(labels) - sum(labels)))
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
            # print("model_path={}, un_model_path={}, \n data_path={}".format(model_path, un_model_path, positive_data_path))
            feature = self.get_differential_data(model_path, un_model_path, positive_data_path).tolist()
            features.append(feature)
            label = [1] * len(feature)
            labels.append(label)
            # 构造反向数据集
            negative_data_path = self.shadow_train_data_paths.format(
                negative_id) if flag == "shadow" else self.target_train_data_paths.format(negative_id)
            # print("model_path={}, un_model_path={}, \n data_path={}".format(model_path, un_model_path,
            #                                                                 negative_data_path))
            feature = self.get_differential_data(model_path, un_model_path, negative_data_path).tolist()
            features.append(feature)
            label = [0] * len(feature)
            labels.append(label)
        features = sum(features, [])
        labels = sum(labels, [])
        print("{}: 正向数据量: {}; 反向数据量: {}".format(flag, sum(labels), len(labels) - sum(labels)))
        return features, labels

    def get_path(self, initial_path):
        files_list = os.listdir(initial_path)
        original_model_path = [i for i in files_list if re.search("original_model.npy", i)][0]
        unlearned_model_path = [i for i in files_list if re.search("_unlearned_model_unid", i)][0]
        unid = unlearned_model_path[len("shadow_unlearned_model_unid_"):-4]
        with open(initial_path + '/' + "negative_id.txt") as f:
            negative_id = int(f.readline())
        return initial_path + original_model_path, initial_path + unlearned_model_path, unid, negative_id

    def get_differential_data(self, model_path, unlearned_path, data_path):
        original_output = self.get_model_output(model_path, data_path)
        unlearned_output = self.get_model_output(unlearned_path, data_path)
        # diff_output = original_output - unlearned_output  # DD
        # print("----------------------------------")
        # print(diff_output)
        diff_output = []
        for o, u in zip(original_output, unlearned_output):
            diff_output.append(((o - u) ** 2).tolist())
        diff_output = torch.tensor(diff_output)
        # print("----------------------------------")
        # print(diff_output)
        return diff_output

    def get_model_output(self, model_path, data_path):
        data = np.load(data_path, allow_pickle=True)
        model = determining_original_model(self.original_model_name, self.dataset_name)
        model_parameter = np.load(model_path, allow_pickle=True).item()
        model.load_state_dict(model_parameter)
        # data = torch.Tensor(data).unsqueeze(1)
        feature = model(data)
        return feature

    def training_attack_model(self, x, y, flag):
        model_flag = "考虑删除数据删除场景" if flag else "不考虑数据删除场景"
        attack_model_path = self.attack_model_path if flag else self.original_attack_model_path
        self.logger.info("{}, 开始训练攻击模型......".format(model_flag))
        # situation = "new" if flag else "base"
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
        # print(test_base_y == test_y)
        test_base_x, test_base_y = np.array(test_base_x), np.array(test_base_y)
        test_x, test_y = np.array(test_x), np.array(test_y)

        base_model_pred = base_model.predict_proba(test_base_x)
        base_model_pred = pd.DataFrame(base_model_pred)

        model_pred = model.predict_proba(test_x)
        model_pred = pd.DataFrame(model_pred)

        diff = model_pred - base_model_pred
        tmp = 0
        for i in range(len(diff)):
            if test_y[i] and diff.loc[i][1] > 0:
                tmp += 1
            elif not test_y[i] and diff.loc[i][1] < 0:
                tmp += 1
        print(tmp / len(test_y))
        diff_1 = (diff[test_y == 1][1] > 0).sum()
        diff_0 = (diff[test_y == 0][1] < 0).sum()
        degcount = (diff_1 + diff_0) / len(test_y)
        diff_1 = diff[test_y == 1][1].sum()
        diff_0 = -diff[test_y == 0][1].sum()
        degrate = (diff_1 + diff_0) / len(test_y)
        return degcount, degrate


if __name__ == "__main__":
    pass

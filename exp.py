import logging
import os.path
import random

import numpy as np
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
        self.attack_model = None
        assert self.unlearning_round <= self.client_number, "去学习个数应该小于客户端个数"
        self.current_time = datetime.now().strftime("%d_%H_%M")
        # self.current_time = "18_01_06"
        file_handler = logging.FileHandler(
            'log/{}_{}_{}_{}.log'.format(self.current_time, self.dataset_name, self.original_model_name,
                                         self.attack_model_name))
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.info(args)
        self.logger.info("Experiment Start!".format())
        self.load_data()

    def load_data(self):
        self.logger.info('loading data')
        self.logger.info('loaded data')


class AttackModelTrainer(Exp):
    def __init__(self, args):
        super(AttackModelTrainer, self).__init__(args)
        self.model = determining_original_model(self.original_model_name)
        # 参与shadow客户端的id
        self.shadow_negative_id = {}
        self.shadow_participants_id = {}
        self.shadow_unlearned_ids = {}

        self.target_negative_id = {}
        self.target_participants_id = {}
        self.target_unlearned_ids = {}

        # shadow数据路径
        self.shadow_train_data_paths = "data/slice/shadow_train_data_{}.npy"
        self.shadow_train_label_paths = "data/slice/shadow_train_label_{}.npy"
        self.shadow_test_data_paths = "data/slice/shadow_test_data_{}.npy"
        self.shadow_test_label_paths = "data/slice/shadow_test_label_{}.npy"
        self.shadow_negative_data_path = "data/slice/shadow_train_data_{}.npy"  # 后面随机抽取id
        self.shadow_all_test_path = ["data/slice/shadow_test_data_{}.npy",
                                     "data/slice/shadow_test_label_{}.npy"]
        # target数据路径
        self.target_train_data_paths = "data/slice/target_train_data_{}.npy"
        self.target_train_label_paths = "data/slice/target_train_label_{}.npy"
        self.target_test_data_paths = "data/slice/target_test_data_{}.npy"
        self.target_test_label_paths = "data/slice/target_test_label_{}.npy"
        self.target_negative_data_path = "data/slice/shadow_train_data_{}.npy"
        self.target_all_test_path = ["data/slice/target_test_data_{}.npy",
                                     "data/slice/target_test_label_{}.npy"]
        # 模型路径
        self.initial_path = "model/{}/".format(self.current_time)
        if not os.path.exists(self.initial_path):
            os.makedirs(self.initial_path)
            self.logger.info("生成模型目录: {} ".format(self.initial_path))
        self.shadow_initial_model_path0 = "shadow_initial_parameters.npy"
        self.shadow_original_model_path0 = "shadow_original_model.npy"
        self.shadow_unlearning_model_path0 = "shadow_unlearned_model_unid_{}.npy"

        self.target_initial_model_path0 = "target_initial_parameters.npy"
        self.target_original_model_path0 = "target_original_model.npy"
        self.target_unlearning_model_path0 = "target_unlearned_model_unid_{}.npy"

        self.attack_model_path = "model/{}/{{}}_attacker.npy".format(self.current_time)

        self.original_number = 2
        self.get_shadow_models(self.original_number)  # 训练多轮shadow模型
        x, y = self.construct_dataset(0)  # 构造训练攻击模型的数据集
        self.training_attack_model(x, y)  # 训练攻击模型

        self.get_target_models(self.original_number)  # 训练target模型
        x, y = self.construct_dataset(1)  # 构造测试攻击模型的数据集
        self.evaluate_attack_model(x, y)

    def get_shadow_models(self, n):
        self.initial_shadow_path = self.initial_path + "shadow_models/"
        if not self.initial_shadow_path:
            os.makedirs(self.initial_shadow_path)
            self.logger.info("成功生成目录: {}".format(self.initial_shadow_path))
        for i in tqdm(range(n)):
            self.shadow_path = self.initial_shadow_path + str(i) + "/"
            if not os.path.exists(self.shadow_path):
                os.makedirs(self.shadow_path)
            print(self.shadow_path)
            self.shadow_initial_model_path = self.shadow_path + self.shadow_initial_model_path0
            self.shadow_original_model_path = self.shadow_path + self.shadow_original_model_path0
            self.shadow_unlearning_model_path = self.shadow_path + self.shadow_unlearning_model_path0
            random_ids = random.sample(range(20), 11)
            self.shadow_negative_id[i] = random_ids[-1]
            self.shadow_participants_id[i] = random_ids[:-1]
            self.shadow_negative_data_path = self.shadow_negative_data_path.format(self.shadow_negative_id[i])
            self.logger.info("shadow: 第 {i} 轮 参与训练的客户端id = {ids} ".format(i=i, ids=self.shadow_participants_id[i]))
            self.logger.info("shadow: 第 {i} 轮 反向数据客户端id = {ids}".format(i=i, ids=self.shadow_negative_id[i]))
            self.training_shadow_model(i)

    def get_target_models(self, n):
        self.initial_target_path = self.initial_path + "target_models/"
        if not self.initial_target_path:
            os.makedirs(self.initial_target_path)
            self.logger.info("成功生成目录: {}".format(self.initial_target_path))
        for i in tqdm(range(n)):
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
            self.target_negative_data_path = self.target_negative_data_path.format(self.target_negative_id[i])
            self.logger.info("target: 第 {i} 轮 参与训练的客户端id = {ids} ".format(i=i, ids=self.target_participants_id[i]))
            self.logger.info("target: 第 {i} 轮 反向数据客户端id = {ids}".format(i=i, ids=self.target_negative_id[i]))
            self.training_target_model(i)

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

    def construct_dataset(self, flag):
        # 构造数据集
        self.logger.info("开始构造攻击模型{}数据集......".format("训练" if not flag else "测试"))
        features = []
        positive_n = 0
        # 构造正向数据集
        for j in range(self.original_number):
            uid = self.shadow_unlearned_ids[j] if not flag else self.target_unlearned_ids[j]
            feature = self.construct_positive_data(j, uid, flag)
            positive_n += len(feature)
            features.append(feature)
        positive_labels = [1] * positive_n
        negative_n = 0
        # 构造反向数据集
        for j in range(self.original_number):
            uid = self.shadow_unlearned_ids[j] if not flag else self.target_unlearned_ids[j]
            feature = self.construct_negative_data(j, uid, flag)
            negative_n += len(feature)
            features.append(feature)
        negative_labels = [0] * negative_n
        features = sum(features, [])
        labels = positive_labels + negative_labels
        print("正向数据量: {}  反向数据量: {}".format(positive_n, negative_n))
        self.logger.info("攻击模型{}完成".format("训练集" if not flag else "测试集"))
        return features, labels

    def construct_positive_data(self, j, uid, flag):
        # 构造uid客户端的的输出，即正向数据
        data_paths = self.shadow_train_data_paths.format(uid) if not flag else self.target_train_data_paths.format(uid)
        data = np.load(data_paths, allow_pickle=True)  # uid客户端的数据，正向数据
        data = torch.Tensor(data).unsqueeze(1)
        positive_feature = self.get_differential_feature(j, uid, data, flag)
        return positive_feature

    def construct_negative_data(self, j, uid, flag):
        negative_id = self.shadow_negative_id[j] if not flag else self.target_negative_id[j]
        data_paths = self.shadow_negative_data_path.format(
            negative_id) if not flag else self.target_negative_data_path.format(negative_id)
        data = np.load(data_paths, allow_pickle=True)  # 反向数据
        data = torch.Tensor(data).unsqueeze(1)
        negative_feature = self.get_differential_feature(j, uid, data, flag)
        return negative_feature

    def get_model_data(self, model_path, data):
        # 得到data输入对应模型的结果
        model_parameter = np.load(model_path, allow_pickle=True).item()
        self.model.load_state_dict(model_parameter)
        preds = self.model(data)
        return preds

    def get_differential_feature(self, j, uid, data, flag):
        #  构造数据在初始模型和uid去学习模型输出的差异
        path = (self.initial_shadow_path if not flag else self.initial_target_path) + str(
            j) + "/"
        original_model = path + self.shadow_original_model_path0 if not flag else \
            path + self.target_original_model_path0
        unlearning_model_paths = path + self.shadow_unlearning_model_path0.format(
            uid) if not flag else path + self.target_unlearning_model_path0.format(uid)
        original_preds = self.get_model_data(original_model, data)
        unleared_preds = self.get_model_data(unlearning_model_paths, data)
        differential_data = original_preds - unleared_preds
        # dataset = torch.utils.data.TensorDataset(torch.Tensor(differential_data), torch.Tensor(label))
        return differential_data.tolist()

    def training_attack_model(self, x, y):
        self.logger.info("开始训练攻击模型......")
        self.attack_model = determine_attack_model(self.attack_model_name)
        attack_model_path = self.attack_model_path.format(self.attack_model_name)
        self.attack_model.train_model(x, y, attack_model_path)
        self.logger.info("{} 攻击模型训练完成！模型保存在 {} ".format(self.attack_model_name, attack_model_path))

    def evaluate_attack_model(self, x, y):
        acc = self.attack_model.test_model_acc(x, y)
        self.logger.info("{} 攻击模型的准确率 acc={}".format(self.attack_model_name, acc))
        auc = self.attack_model.test_model_auc(x, y)
        self.logger.info("{} 模型 auc = {}".format(self.attack_model_name, auc))


if __name__ == "__main__":
    args = get_args()
    AttackModelTrainer(args)

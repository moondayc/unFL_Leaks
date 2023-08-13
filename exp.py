import logging
import numpy as np
import torch.utils.data
from tqdm import tqdm
from FL import FederatedTrainer
from Model import LeNet
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
        self.attack_model_name = args["attack_model_name"]
        self.attack_model = None
        assert self.unlearning_round <= self.client_number, "去学习个数应该小于客户端个数"
        current_time = datetime.now().strftime("%d_%H-%M")
        file_handler = logging.FileHandler(
            'log/{}_{}_{}_{}.log'.format(current_time, self.dataset_name, self.original_model_name,
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
        # shadow数据路径
        self.shadow_train_data_paths = "data/shadow_train_data_part_{}.npy"
        self.shadow_train_label_paths = "data/shadow_train_label_part_{}.npy"
        self.shadow_test_data_paths = "data/shadow_test_data_part_{}.npy"
        self.shadow_test_label_paths = "data/shadow_test_label_part_{}.npy"
        self.shadow_negative_data_path = "data/shadow_train_data_part_10.npy"
        self.shadow_all_test_path = ["data/Completed_dataset/shadow_test_data.npy",
                                     "data/Completed_dataset/shadow_test_label.npy"]
        # target数据路径
        self.target_train_data_paths = "data/target_train_data_part_{}.npy"
        self.target_train_label_paths = "data/target_train_label_part_{}.npy"
        self.target_test_data_paths = "data/target_test_data_part_{}.npy"
        self.target_test_label_paths = "data/target_test_label_part_{}.npy"
        self.target_negative_data_path = "data/target_train_data_part_10.npy"
        self.target_all_test_path = ["data/Completed_dataset/target_test_data.npy",
                                     "data/Completed_dataset/target_test_label.npy"]
        # 模型路径
        self.shadow_initial_model_path = "model/shadow_initial_parameters.npy"
        self.shadow_original_model_path = "model/shadow_original_model.npy"
        self.shadow_unlearning_model_path = "model/shadow_unlearned_model_unid_{}.npy"

        self.target_initial_model_path = "model/target_initial_parameters.npy"
        self.target_original_model_path = "model/target_original_model.npy"
        self.target_unlearning_model_path = "model/target_unlearned_model_unid_{}.npy"

        self.attack_model_path = "model/{}_attacker.npy"

        self.training_shadow_model()  # 训练shadow模型
        x, y = self.construct_dataset(0)  # 构造训练攻击模型的数据集
        self.training_attack_model(x, y)  # 训练攻击模型

        self.training_target_model()  # 训练target模型
        x, y = self.construct_dataset(1)  # 构造测试攻击模型的数据集
        self.evaluate_attack_model(x, y)

    def training_shadow_model(self):  # 训练shadow模型
        initial_parameters = {}
        if not initial_parameters:
            for key, var in self.model.state_dict().items():
                initial_parameters[key] = var.clone()
        np.save(self.shadow_initial_model_path, initial_parameters)
        self.logger.info("shadow 全局变量初始化完成")
        ftrainer = FederatedTrainer(self.client_number, self.original_model_name, self.shadow_initial_model_path)
        data_path = [self.shadow_train_data_paths, self.shadow_train_label_paths, self.shadow_test_data_paths,
                     self.shadow_test_label_paths, self.shadow_all_test_path]
        ftrainer.training(self.max_agg_round, self.shadow_original_model_path, self.local_epoch, self.local_batch_size,
                          data_path)
        self.logger.info('shadow初始模型训练完成')

        self.logger.info('开始训练shadow去学习模型.....')
        for un in tqdm(range(self.unlearning_round)):
            self.logger.info("unlearning id = {}".format(un))
            unlearning_model_path = self.shadow_unlearning_model_path.format(un)
            ftrainer.training(self.max_agg_round, unlearning_model_path, self.local_epoch, self.local_batch_size,
                              data_path, [un])
        self.logger.info('shadow 模型训练完成')

    def construct_dataset(self, flag):
        # 构造数据集
        self.logger.info("开始{}攻击模型数据集......".format("构造" if not flag else "测试"))
        features = []
        positive_n = 0
        # 构造正向数据集
        for uid in range(self.unlearning_round):
            feature = self.construct_positive_data(uid, flag)
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
        self.logger.info("攻击模型{}完成".format("训练集" if not flag else "测试集"))
        return features, labels

    def construct_positive_data(self, uid, flag):
        # 构造uid客户端的的输出，即正向数据
        data_paths = self.shadow_train_data_paths.format(uid) if not flag else self.target_train_data_paths.format(uid)
        data = np.load(data_paths, allow_pickle=True)  # uid客户端的数据，正向数据
        data = torch.Tensor(data).unsqueeze(1)
        positive_feature = self.get_differential_feature(uid, data, flag)
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

    def training_attack_model(self, x, y):
        self.logger.info("开始训练攻击模型......")
        self.attack_model = determine_attack_model(self.attack_model_name)
        attack_model_path = self.attack_model_path.format(self.attack_model_name)
        self.attack_model.train_model(x, y, attack_model_path)
        self.logger.info("{} 攻击模型训练完成！模型保存在 {} ".format(self.attack_model_name, attack_model_path))

    def training_target_model(self):
        initial_parameters = {}
        if not initial_parameters:
            for key, var in self.model.state_dict().items():
                initial_parameters[key] = var.clone()
        np.save(self.target_initial_model_path, initial_parameters)
        self.logger.info("target 全局变量初始化完成")
        self.logger.info("开始训练target模型......")
        ttrainer = FederatedTrainer(self.client_number, self.original_model_name, self.target_initial_model_path)
        data_path = [self.target_train_data_paths, self.target_train_label_paths, self.target_test_data_paths,
                     self.target_test_label_paths, self.target_all_test_path]
        ttrainer.training(self.max_agg_round, self.target_original_model_path, self.local_epoch, self.local_batch_size,
                          data_path)
        self.logger.info('target初始模型训练完成')

        self.logger.info('开始训练target去学习模型.....')
        for un in tqdm(range(self.unlearning_round)):
            self.logger.info("unlearning id = {}".format(un))
            unlearning_model_path = self.target_unlearning_model_path.format(un)
            ttrainer.training(self.max_agg_round, unlearning_model_path, self.local_epoch, self.local_batch_size,
                              data_path, [un])
        self.logger.info('shadow 模型训练完成')

    def evaluate_attack_model(self, x, y):
        acc = self.attack_model.test_model_acc(x, y)
        print("{} 攻击模型的准确率 acc={}".format(self.attack_model_name, acc))
        auc = self.attack_model.test_model_auc(x, y)
        print("{} 模型 auc = {}".format(self.attack_model_name, auc))

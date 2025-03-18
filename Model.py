import pickle
import random

import joblib
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch.nn import Module
import torch
from torch import nn
import torch.nn.functional as F
"""
    客户端和攻击的模型选择
"""

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, in_dim=1, out_dim=10):
        # for mnist,in_dim=1,nn.Linear(28 * 12 * 12, 128)
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=(3, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(1, 1))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 14, 128)
        self.fc2 = nn.Linear(128, out_dim)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # temperature = 2  为什么要除以2
        # x /= temperature
        # output = F.log_softmax(x, dim=1)
        return x


class CNNCifar(nn.Module):
    def __init__(self, n, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(n, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LR:
    def __init__(self, dataset_name):
        self.model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=1)
        self.dataset_name = dataset_name
        self.initialize_parameters()
        with open('data/{}/scaler.pkl'.format(self.dataset_name), 'rb') as file:
            self.scaler = pickle.load(file)
        self.flag = 0  # 表示是否使用了上次的训练结果

    def __call__(self, x):
        # return self.predict_proba(self.scaler.transform(x))
        # print(x)
        # print(self.scaler.transform(x))
        return self.predict_proba(self.scaler.transform(x))

    def train_model(self, train_x, train_y, save_name=None):
        if self.flag:
            train_y = np.concatenate((self.predict(self.train_x), train_y), axis=0)
            train_x = np.concatenate((train_x, train_x), axis=0)
        # self.scaler = preprocessing.StandardScaler().fit(train_x)
        self.model.fit(self.scaler.transform(train_x), train_y)
        if save_name:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def state_dict(self):
        params = {
            "coef": self.model.coef_,
            "intercept": self.model.intercept_,
            'classes_': self.model.classes_
        }
        return params

    # set_params
    def load_state_dict(self, params):
        self.model.coef_ = params["coef"]
        self.model.intercept_ = params["intercept"]
        self.model.classes_ = params['classes_']
        self.flag = 1

    def initialize_parameters(self):
        if self.dataset_name == "adult":
            self.model.coef_ = np.random.rand(1, 14)
            self.model.intercept_ = np.random.rand(1)
            self.model.classes_ = np.array([0, 1], dtype=np.uint32)
        elif self.dataset_name == "accident":
            self.model.coef_ = np.random.rand(4, 29)
            self.model.intercept_ = np.random.rand(4)
            self.model.classes_ = np.array([0, 1, 2, 3], dtype=np.uint32)
        elif self.dataset_name == "insta_ny":
            self.model.coef_ = np.random.rand(9, 168)
            self.model.intercept_ = np.random.rand(9)
            self.model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
        elif self.dataset_name == "insta_la":
            self.model.coef_ = np.random.rand(9, 168)
            self.model.intercept_ = np.random.rand(9)
            self.model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
        else:
            raise "数据集在LR中没有初始化"

    def predict_proba(self, test_x):
        return self.model.predict_proba(self.scaler.transform(test_x))

    def predict(self, test_x):
        return self.model.predict(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(self.scaler.transform(test_x))
        return accuracy_score(test_y, pred_y) * 100

    # def test_model_auc(self, test_x, test_y):
    #     pred_y = self.model.predict_proba(self.scaler.transform(test_x))
    #     # return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
    #     return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC


class LR_without_scaler:
    def __init__(self, dataset_name):
        self.model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=1)
        self.dataset_name = dataset_name
        self.initialize_parameters()
        self.flag = 0  # 表示是否使用了上次的训练结果

    def __call__(self, x):
        return self.predict_proba(x)

    def train_model(self, train_x, train_y, save_name=None):
        if self.flag:
            train_y = np.concatenate((self.predict(train_x), train_y), axis=0)
            train_x = np.concatenate((train_x, train_x), axis=0)
        # self.scaler = preprocessing.StandardScaler().fit(train_x)
        self.model.fit(train_x, train_y)
        if save_name:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def state_dict(self):
        params = {
            "coef": self.model.coef_,
            "intercept": self.model.intercept_,
            'classes_': self.model.classes_
        }
        return params

    # set_params
    def load_state_dict(self, params):
        self.model.coef_ = params["coef"]
        self.model.intercept_ = params["intercept"]
        self.model.classes_ = params['classes_']
        self.flag = 1

    def initialize_parameters(self):
        if self.dataset_name == "adult":
            self.model.coef_ = np.random.rand(1, 14)
            self.model.intercept_ = np.random.rand(1)
            self.model.classes_ = np.array([0, 1], dtype=np.uint32)
        elif self.dataset_name == "accident":
            self.model.coef_ = np.random.rand(4, 29)
            self.model.intercept_ = np.random.rand(4)
            self.model.classes_ = np.array([0, 1, 2, 3], dtype=np.uint32)
        elif self.dataset_name == "insta_ny":
            self.model.coef_ = np.random.rand(9, 168)
            self.model.intercept_ = np.random.rand(9)
            self.model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
        elif self.dataset_name == "insta_la":
            self.model.coef_ = np.random.rand(9, 168)
            self.model.intercept_ = np.random.rand(9)
            self.model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
        else:
            raise "数据集在LR中没有初始化"

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def predict(self, test_x):
        return self.model.predict(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y) * 100


# ----------------------------------- 攻击模型
class DT:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class RF:
    def __init__(self, min_samples_leaf=30):
        self.model = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=min_samples_leaf)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class MLP:
    def __init__(self):
        self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class LR_2:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=1)
        self.initialize_parameters()
        self.scaler = preprocessing.StandardScaler()
        # scaler_path = 'data/{}/{}_{}_attacker_scaler.pkl'.format(dataset_name, situation, original_model)
        # print("scaler_path: {}".format(scaler_path))
        # with open(scaler_path, 'rb') as file:
        #     self.scaler = pickle.load(file)

    # def __call__(self, x):
    #     print(x)
    #     print(self.scaler.transform(x))
    #     return self.predict_proba(self.scaler.transform(x))

    def train_model(self, train_x, train_y, save_name=None):
        self.scaler.fit(train_x)
        self.model.fit(self.scaler.transform(train_x), train_y)
        if save_name:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def state_dict(self):
        params = {
            "coef": self.model.coef_,
            "intercept": self.model.intercept_,
            'classes_': self.model.classes_
        }
        return params

    # set_params
    def load_state_dict(self, params):
        self.model.coef_ = params["coef"]
        self.model.intercept_ = params["intercept"]
        self.model.classes_ = params['classes_']

    def initialize_parameters(self):
        self.model.coef_ = np.random.rand(1, 2)
        self.model.intercept_ = np.random.rand(1)
        self.model.classes_ = np.array([0, 1], dtype=np.int64)

    def predict_proba(self, test_x):
        return self.model.predict_proba(self.scaler.transform(test_x))

    def predict(self, test_x):
        return self.model.predict(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(self.scaler.transform(test_x))
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
        # return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC


class LR_2_without_scaler:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=1)
        self.initialize_parameters()

    def __call__(self, x):
        return self.predict_proba(x)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def state_dict(self):
        params = {
            "coef": self.model.coef_,
            "intercept": self.model.intercept_,
            'classes_': self.model.classes_
        }
        return params

    # set_params
    def load_state_dict(self, params):
        self.model.coef_ = params["coef"]
        self.model.intercept_ = params["intercept"]
        self.model.classes_ = params['classes_']

    def initialize_parameters(self):
        self.model.coef_ = np.random.rand(1, 2)
        self.model.intercept_ = np.random.rand(1)
        self.model.classes_ = np.array([0, 1], dtype=np.int64)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def predict(self, test_x):
        return self.model.predict(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])

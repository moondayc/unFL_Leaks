import pickle

import numpy as np
import pandas as pd
import torch

from Model import DT, LR, MLP, RF, LR_2


def calculate_DegCount_and_DegRate(self, test_base_x, test_base_y, base_model, test_x, test_y, model):
    assert test_base_y == test_y, "两个攻击模型的测试数据必须相同"
    print(test_base_y == test_y)
    test_base_x, test_base_y = np.array(test_base_x), np.array(test_base_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    base_model_pred = base_model.predict_proba(test_base_x)
    base_model_pred = pd.DataFrame(base_model_pred)

    model_pred = model.predict_proba(test_x)
    model_pred = pd.DataFrame(model_pred)

    diff = model_pred - base_model_pred
    # tmp = 0
    # for i in range(len(diff)):
    #     if test_y[i] and diff.loc[i][1] > 0:
    #         tmp += 1
    #     elif not test_y[i] and diff.loc[i][1] < 0:
    #         tmp += 1
    # print(tmp/len(test_y))
    diff_1 = (diff[test_y == 1][1] > 0).sum()
    diff_0 = (diff[test_y == 0][1] < 0).sum()
    degcount = (diff_1 + diff_0) / len(test_y)
    diff_1 = diff[test_y == 1][1].sum()
    diff_0 = -diff[test_y == 0][1].sum()
    degrate = (diff_1 + diff_0) / len(test_y)
    return degcount, degrate


if __name__ == "__main__":
    # model = LR("accident")
    # x = np.load("data/slice/accident/shadow_train_data_13.npy", allow_pickle=True)
    # y = np.load("data/slice/accident/shadow_train_label_13.npy", allow_pickle=True)
    # print(set(y))
    # print("训练之前")
    # print(model.state_dict())
    # model.train_model(x, y, "1.npy")
    # test_x = np.load("data/slice/accident/shadow_test_data_13.npy", allow_pickle=True)
    # test_y = np.load("data/slice/accident/shadow_test_label_13.npy", allow_pickle=True)
    #
    # pred = model.predict(test_x)
    #
    # print("训练之后")
    # print(model.state_dict())
    # acc = model.test_model_acc(x, y)
    # print(acc)
    # acc = model.test_model_acc(test_x, test_y)
    # print(acc)
    # ---------------------
    # for i in range(20):
    #     x = np.load("data/slice/insta_ny/shadow_test_data_{}.npy".format(i), allow_pickle=True)
    #     print("shadow 客户端{}训练数据：{}".format(i, len(x)))
    # for i in range(20):
    #     y = np.load("data/slice/insta_la/target_train_label_{}.npy".format(i), allow_pickle=True)
    #     if len(set(y)) != 9:
    #         print(i)
    # x = np.load("data/slice/adult/shadow_test_data_{}.npy".format(0), allow_pickle=True)
    # x0 = x[:2]
    # print(x0)
    # x0 = [[-0.5, 0.5], [0.0005, -0.0005]]
    # model = LR_2()
    # scaler_path = 'data/{}/{}_{}_attacker_scaler.pkl'.format("adult", "new", "LR")
    # print("scaler_path: {}".format(scaler_path))
    # with open(scaler_path, 'rb') as file:
    #     scaler = pickle.load(file)
    # a = scaler.transform(x0)
    # print(a)
    # scaler_path = 'data/{}/{}_{}_attacker_scaler.pkl'.format("adult", "base", "LR")
    # print("scaler_path: {}".format(scaler_path))
    # with open(scaler_path, 'rb') as file:
    #     scaler = pickle.load(file)
    # a = scaler.transform(x0)
    # print(a)
    # print(model(x0))
    # aver = (
    #         -0.03777722641825676 - 0.04382312670350075 - 0.03692822903394699 - 0.042400676757097244 - 0.04816630482673645 - 0.035528428852558136 - 0.03994421660900116 - 0.05780096352100372 - 0.022186817601323128 - 0.08588425070047379)
    # print(aver + 0.01351505 * 10)
    # print(aver / 10)

    original_output = torch.Tensor([[1, 2, 3], [1, 2, 3]])
    unlearned_output = torch.Tensor([[3, 1, 2], [3, 1, 2]])
    diff_output = []

    for o, u in zip(original_output, unlearned_output):
        diff_output.append((o - u) ** 2)

    print(diff_output)
    diff_output = torch.cat(diff_output, dim=0)

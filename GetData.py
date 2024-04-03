import os
import gzip
import pickle
import struct
import numpy as np
import torch
import torchvision
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

dataset_name = "mnist"
# ["accident", "adult", "adult_without","insta_la", "insta_ny", "mnist"]
if not os.path.exists("data/slice"):
    os.makedirs("data/slice")
if dataset_name == "mnist":
    if not os.path.exists("data/slice/mnist"):
        os.makedirs("data/slice/mnist")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='data/MNIST', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Load the test set
    testset = torchvision.datasets.MNIST(root='data/MNIST', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    train_data = trainset.data.numpy()
    train_labels = trainset.targets.numpy()
    test_data = testset.data.numpy()
    test_labels = testset.targets.numpy()
elif dataset_name == "adult":
    if not os.path.exists("data/slice/adult"):
        os.makedirs("data/slice/adult")
    df = pickle.load(open("data/adult/adult", 'rb'))
    df = df[['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
             'occupation', 'relationship', 'marital-status', 'race', 'gender', 'capital-gain',
             'capital-loss', 'hours-per-week', 'native-country', 'income']]
    # print(df)
    data = df[['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
               'occupation', 'relationship', 'marital-status', 'race', 'gender', 'capital-gain',
               'capital-loss', 'hours-per-week', 'native-country']].values
    label = df['income'].values
elif dataset_name == "accident":
    if not os.path.exists("data/slice/{}".format(dataset_name)):
        os.makedirs("data/slice/{}".format(dataset_name))
    df = pickle.load(open("data/accident/accident", 'rb'))
    # 3-class balanced
    df = df[['Source', 'TMC', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
             'Side', 'County', 'State', 'Timezone', 'Airport_Code', 'Temperature(F)',
             'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
             'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)',
             'Weather_Condition', 'Amenity', 'Crossing', 'Junction', 'Railway',
             'Station', 'Traffic_Signal', 'Sunrise_Sunset', 'Civil_Twilight',
             'Nautical_Twilight', 'Astronomical_Twilight', 'Severity']]
    df['Severity'] = df['Severity'].replace({1: 0, 2: 1, 3: 2, 4: 3,})
    # 打乱 DataFrame 的行
    df = df.sample(frac=1, random_state=42)
    # 重置索引
    df = df.reset_index(drop=True)
    data = df[['Source', 'TMC', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
               'Side', 'County', 'State', 'Timezone', 'Airport_Code', 'Temperature(F)',
               'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
               'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)',
               'Weather_Condition', 'Amenity', 'Crossing', 'Junction', 'Railway',
               'Station', 'Traffic_Signal', 'Sunrise_Sunset', 'Civil_Twilight',
               'Nautical_Twilight', 'Astronomical_Twilight', ]].values
    label = df['Severity'].values

    train_data, train_labels = data[:int(len(data) * 0.8)], label[:int(len(label) * 0.8)]
    test_data, test_labels = data[int(len(data) * 0.8):], label[int(len(label) * 0.8):]
    scaler = preprocessing.StandardScaler().fit(data)
    with open('data/{}/scaler.pkl'.format(dataset_name), 'wb') as file:
        pickle.dump(scaler, file)

elif dataset_name == "insta_ny":
    if not os.path.exists("data/slice/{}".format(dataset_name)):
        os.makedirs("data/slice/{}".format(dataset_name))
    df = pickle.load(open("data/insta_ny/insta_ny", 'rb'))
    name = df.columns
    #print(name)
    data = df[name[:-1]].values
    label = df["catid"].values
    scaler = preprocessing.StandardScaler().fit(data)
    with open('data/{}/scaler.pkl'.format(dataset_name), 'wb') as file:
        pickle.dump(scaler, file)
    #print(df)
    train_data, train_labels = data[:int(len(data) * 0.8)], label[:int(len(label) * 0.8)]
    test_data, test_labels = data[int(len(data) * 0.8):], label[int(len(label) * 0.8):]

elif dataset_name == "insta_la":
    if not os.path.exists("data/slice/{}".format(dataset_name)):
        os.makedirs("data/slice/{}".format(dataset_name))
    df = pickle.load(open("data/insta_la/insta_la", 'rb'))
    # 打乱 DataFrame 的行
    df = df.sample(frac=1, random_state=42)
    # 重置索引
    df = df.reset_index(drop=True)
    name = df.columns
    data = df[name[:-1]].values
    label = df["catid"].values
    scaler = preprocessing.StandardScaler().fit(data)
    with open('data/{}/scaler.pkl'.format(dataset_name), 'wb') as file:
        pickle.dump(scaler, file)
    print(df)
    train_data, train_labels = data[:int(len(data) * 0.8)], label[:int(len(label) * 0.8)]
    test_data, test_labels = data[int(len(data) * 0.8):], label[int(len(label) * 0.8):]

print("训练数据总数：{}".format(len(train_data)))
print("测试数据总数：{}".format(len(test_data)))
# 分割数据
# 将训练集分成 n 个部分
train_n = 40
test_n = 40
train_data_parts = np.array_split(train_data, train_n)
train_labels_parts = np.array_split(train_labels, train_n)

test_data_parts = np.array_split(test_data, test_n)
test_labels_parts = np.array_split(test_labels, test_n)

# 先保存target的数据
# 正向数据
for i in range(20):
    np.save("data/slice/{}/target_train_data_{}.npy".format(dataset_name, i), train_data_parts[i])
    np.save("data/slice/{}/target_train_label_{}.npy".format(dataset_name, i), train_labels_parts[i])

    np.save("data/slice/{}/target_test_data_{}.npy".format(dataset_name, i), test_data_parts[i])
    np.save("data/slice/{}/target_test_label_{}.npy".format(dataset_name, i), test_labels_parts[i])

    print("target: 客户端{}的训练数据量:{}".format(i, len(train_data_parts[i])))
    print("target: 客户端{}的测试数据量:{}".format(i, len(test_data_parts[i])))

# 下标11到29作为shadow的训练数据
for i in range(20, 40):
    np.save("data/slice/{}/shadow_train_data_{}.npy".format(dataset_name, i - 20), train_data_parts[i])
    np.save("data/slice/{}/shadow_train_label_{}.npy".format(dataset_name, i - 20), train_labels_parts[i])

    np.save("data/slice/{}/shadow_test_data_{}.npy".format(dataset_name, i - 20), test_data_parts[i])
    np.save("data/slice/{}/shadow_test_label_{}.npy".format(dataset_name, i - 20), test_labels_parts[i])

    print("shadow: 客户端{}的训练数据量:{}".format(i - 20, len(train_data_parts[i])))
    print("shadow: 客户端{}的测试数据量:{}".format(i - 20, len(test_data_parts[i])))

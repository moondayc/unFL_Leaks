import os
import gzip
import pickle
import struct
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

# def load_mnist(path, kind='train'):
#     """Load MNIST data"""
#     labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
#     images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
#     with gzip.open(labels_path, 'rb') as lbpath:
#         magic, n = struct.unpack('>II', lbpath.read(8))
#         labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
#     with gzip.open(images_path, 'rb') as imgpath:
#         magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
#         images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 1, 28, 28)
#     return images, labels
#
# # 加载训练集和测试集
# train_images, train_labels = load_mnist('data/MNIST', 'train')
# test_images, test_labels = load_mnist('data/MNIST', 'test')
#
# # 将数据转换成PyTorch的Tensor
# train_images = torch.tensor(train_images, dtype=torch.float32)
# train_labels = torch.tensor(train_labels, dtype=torch.long)
# test_images = torch.tensor(test_images, dtype=torch.float32)
# test_labels = torch.tensor(test_labels, dtype=torch.long)
#
# # 创建TensorDataset对象
# train_dataset = TensorDataset(train_images, train_labels)
# test_dataset = TensorDataset(test_images, test_labels)
#
# # 创建DataLoader对象
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLdataoader(test_dataset, batch_size=64, shuffle=False)
#
# with open('train_loader.pkl', 'wb') as f:
#     pickle.dump(train_loader, f)
#
# with open('test_loader.pkl', 'wb') as f:
#     pickle.dump(test_loader, f)

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# # Load the training set
trainset = torchvision.datasets.MNIST(root='data/MNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Load the test set
testset = torchvision.datasets.MNIST(root='data/MNIST', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print("训练数据总数：{}".format(len(trainset)))
print("测试数据总数：{}".format(len(testset)))
# 分割数据
# 将训练集分成 n 个部分
train_n = 40
test_n = 20
train_data = trainset.data.numpy()
train_labels = trainset.targets.numpy()
# train_data_parts = np.array_split(train_data, train_n)
# train_labels_parts = np.array_split(train_labels, train_n)

test_data = testset.data.numpy()
test_labels = testset.targets.numpy()
test_data_parts = np.array_split(test_data, test_n)
test_labels_parts = np.array_split(test_labels, test_n)

# 先将数据分成四个部分
train_data_parts = np.array_split(train_data, 4)
train_labels_parts = np.array_split(train_labels, 4)

# 将前两个保存成反向数据
np.save("data/slice/shadow_negative_data.npy", train_data_parts[0])
print("shadow:反向数据的数量：{}".format(len(train_data_parts[0])))
np.save("data/slice/target_negative_data.npy", train_data_parts[1])
print("target:反向数据的数量：{}".format(len(train_data_parts[1])))

np.save("data/slice/shadow_negative_label.npy", train_labels_parts[0])
np.save("data/slice/target_negative_label.npy", train_labels_parts[1])

# 将测试集分成的20部分

# 第三份分成shadow的训练集，和前半部分的测试集
shadow_train_data_parts = np.array_split(train_data_parts[2], 10)
shadow_train_labels_parts = np.array_split(train_labels_parts[2], 10)
for i in range(10):
    np.save("data/slice/shadow_train_data_{}.npy".format(i), shadow_train_data_parts[i])
    np.save("data/slice/shadow_train_label_{}.npy".format(i), shadow_train_labels_parts[i])

    np.save("data/slice/shadow_test_data_{}.npy".format(i), test_data_parts[i])
    np.save("data/slice/shadow_test_label_{}.npy".format(i), test_labels_parts[i])

    print("shadow: 客户端{}的训练数据量:{}".format(i, len(shadow_train_data_parts[i])))
    print("shadow: 客户端{}的测试数据量:{}".format(i, len(test_data_parts[i])))


# 第四份分成target的训练集
target_train_data_parts = np.array_split(train_data_parts[3], 10)
target_train_labels_parts = np.array_split(train_labels_parts[3], 10)
for i in range(10):
    np.save("data/slice/target_train_data_{}.npy".format(i), target_train_data_parts[i])
    np.save("data/slice/target_train_label_{}.npy".format(i), target_train_labels_parts[i])

    np.save("data/slice/target_test_data_{}.npy".format(i), test_data_parts[i+10])
    np.save("data/slice/target_test_label_{}.npy".format(i), test_labels_parts[i+10])

    print("target: 客户端{}的训练数据量:{}".format(i, len(target_train_data_parts[i])))
    print("target: 客户端{}的测试数据量:{}".format(i, len(test_data_parts[i+10])))

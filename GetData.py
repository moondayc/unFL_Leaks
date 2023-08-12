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

# Load the training set
trainset = torchvision.datasets.MNIST(root='data/MNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Load the test set
testset = torchvision.datasets.MNIST(root='data/MNIST', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print("训练数据总数：{}".format(len(trainset)))
print("测试数据总数：{}".format(len(testset)))
# 分割数据
# 将训练集分成 n 个部分
train_n = 22
test_n = 20
train_data = trainset.data.numpy()
train_labels = trainset.targets.numpy()
train_data_parts = np.array_split(train_data, train_n)
train_labels_parts = np.array_split(train_labels, train_n)

test_data = testset.data.numpy()
test_labels = testset.targets.numpy()
test_data_parts = np.array_split(test_data, test_n)
test_labels_parts = np.array_split(train_labels, test_n)
# for i in range(n):
#     np.save(f'data/Completed_dataset/train_data.npy', train_data_parts[i])
#     np.save(f'data/Completed_dataset/train_labels.npy', train_labels_parts[i])
#     np.save(f'data/Completed_dataset/test_data.npy', train_data_parts[i])
#     np.save(f'data/Completed_dataset/test_labels.npy', train_labels_parts[i])

for i in range(11):
    np.save(f'data/shadow_train_data_part_{i}.npy', train_data_parts[i])
    np.save(f'data/shadow_train_label_part_{i}.npy', train_labels_parts[i])
for i in range(11, train_n):
    np.save(f'data/target_train_data_part_{i-11}.npy', train_data_parts[i])
    np.save(f'data/target_train_label_part_{i-11}.npy', train_labels_parts[i])
for i in range(10):
    np.save(f'data/shadow_test_data_part_{i}.npy', train_data_parts[i])
    np.save(f'data/shadow_test_label_part_{i}.npy', train_labels_parts[i])
for i in range(10, train_n):
    np.save(f'data/target_test_data_part_{i-10}.npy', train_data_parts[i])
    np.save(f'data/target_test_label_part_{i-10}.npy', train_labels_parts[i])

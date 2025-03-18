import numpy as np
"""
    模拟安全聚合过程
"""

class SecAgg:
    def __init__(self, decimal_places):
        self.decimal_places = decimal_places
        self.agg = None
        self.dim = 0
        self.number_of_client = 0

    def add(self, data): #模拟服务器接收客户端模型过程
        data = np.array([int(i*10**self.decimal_places) / (10**self.decimal_places) for i in data])
        if not self.dim:
            self.dim = len(data)
            self.agg = np.zeros((self.dim,))
        self.agg += data
        self.number_of_client += 1

    def summary(self):  # 模拟聚合过程
        return self.agg

    def average(self):
        average = self.agg / self.number_of_client
        average = [int(i*10**self.decimal_places) / (10**self.decimal_places) for i in average]
        return average

if __name__ == "__main__":
    pass
    # Pro = SecAgg(1)
    # a = [0.100000001, 0.9]
    # b = [0.123456098, 0.12]
    # Pro.add(a)
    # Pro.add(b)
    # print(Pro.average())


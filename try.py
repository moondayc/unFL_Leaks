import numpy as np
import pandas as pd

from Model import DT


def calculate_DegCount_and_DegRate(test_x, test_y, test_model, base_model):
    tree1_pred = test_model.predict_proba(test_x)
    tree2_pred = base_model.predict_proba(test_x)
    tree1_pred = pd.DataFrame(tree1_pred)
    tree2_pred = pd.DataFrame(tree2_pred)
    diff = tree1_pred - tree2_pred
    diff_1 = (diff[test_y == 1][1] > 0).sum()
    diff_0 = (diff[test_y == 0][1] < 0).sum()
    degcount = (diff_1+diff_0)/len(test_y)
    diff_1 = diff[test_y == 1][1].sum()
    diff_0 = -diff[test_y == 0][1].sum()
    degrate = (diff_1+diff_0)/len(test_y)
    print(degcount, degrate)
    return degcount, degrate


    # print("tree1_pred")
    # print(tree1_pred)
    # test_x_1 = test_x[test_y == 1]
    # print(test_x_1)


    pass


if __name__ == "__main__":
    tree1 = DT()
    x = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    y = [1, 1, 1, 0, 0, 0]
    tree1.train_model(x, y, "1.npy")
    pred = tree1.predict_proba([[1, 1, 1],
                                [0, 0, 0]])
    tree2 = DT()
    x = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    y = [0, 0, 0, 1, 1, 1]
    tree2.train_model(x, y, "1.npy")
    pred = tree2.predict_proba([[1, 1, 1],
                                [0, 0, 0]])
    kx = np.array([[1, 1, 1],
          [0, 0, 0],
          [0, 0, 0],
          [1, 1, 1]])
    ky = np.array([1, 0, 0, 1])
    calculate_DegCount_and_DegRate(kx, ky, tree1, tree2)

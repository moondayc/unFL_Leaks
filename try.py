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


if __name__ == "__main__":
    model = DT()
    x = [
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ]
    y = ["花", "花", "草", "草"]
    model.train_model(x, y, "1.npy")
    k = [
        [1, 1, 0],
        [0, 0, 1]
    ]
    p = model.predict_proba(k)
    print(p)

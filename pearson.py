from math import sqrt
import pandas as pd
import numpy as np
import pickle


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den


if __name__ == '__main__':
    raw_data = pd.read_hdf("./data/pems-bay.h5")
    raw_data = raw_data.values
    data = np.zeros(raw_data.shape)
    scale = np.ones(data.shape[1])
    for i in range(data.shape[1]):
        scale[i] = np.max(np.abs(raw_data[:, i]))
        data[:, i] = raw_data[:, i] / np.max(np.abs(raw_data[:, i]))
    pearson_mx = np.zeros((data.shape[1], data.shape[1]))
    for u_idx in range(data.shape[1]):
        print(u_idx)
        print("---------------")
        for v_idx in range(data.shape[1]):
            print(u_idx, ":", v_idx)
            seq_a = data[:, u_idx]
            seq_b = data[:, v_idx]
            corr = corrcoef(seq_a, seq_b)
            pearson_mx[u_idx][v_idx] = corr
    with open("./data/PEMS-BAY/vitrual_graph.pkl", "wb") as f:
        pickle.dump(pearson_mx, f)

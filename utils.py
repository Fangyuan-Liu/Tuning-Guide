import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import random

random_state = 1111
scaler = MinMaxScaler(feature_range=(-1, 1))

def min_max_normalization(np_array):
    min_max_scaler = MinMaxScaler()
    ret = min_max_scaler.fit_transform(np_array)
    return ret


def dataset_split_kfold(X, y, k=5):
    """
    ref: https://blog.csdn.net/u010986753/article/details/98069124
    用于划分数据集
    :return:
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    kf_datasets = []
    for train_index, test_index in kf.split(X):
        x_train = [X[i] for i in train_index]
        x_test = [X[i] for i in test_index]
        kf_datasets.append({
            "X_train": x_train,
            "X_test": x_test,
            "y_train": y[train_index],
            "y_test": y[test_index],
            "train_index": train_index,
            "test_index": test_index
        })
    return kf_datasets


def dataset_split(X, y):
    kf_datasets = []
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    kf_datasets.append({
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    })
    return kf_datasets


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

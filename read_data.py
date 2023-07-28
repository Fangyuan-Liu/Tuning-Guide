import pandas as pd
import numpy as np
from utils import *

# df = pd.read_csv("./all_data/features_all_data_0725.csv", encoding="GB18030")
# y = df.loc[:, 'label']
# X = df.loc[:, df.columns[8:]].fillna(value=0)  # 缺失值填充为0
# # df.columns[8:]

class ReadData:
    def __init__(self, csv_path, vec_path=None):
        self.df = pd.read_csv(csv_path, encoding="GB18030")
        self.vec_path = vec_path
        self.X = None
        self.y = None

    def run(self, method, mode="train", norm=True):
        if method == "feature":
            self.read_data_features()
        elif method == "sentence_vector" and type(self.vec_path) is str:
            self.read_data_sentence_vector()
        elif method == "word_vector" and type(self.vec_path) is list:
            self.read_data_word_vector()
        elif method == "text":
            norm = False
            self.read_data_text()
        else:
            raise "Error! Please give a valid method for reading data!"

        if norm:
            self.X = min_max_normalization(self.X)

        if mode == "train":
            # normalized_X = scaler.fit_transform(X.reshape(-1, 1))
            self.y = np.array(self.df.loc[:, 'label'])
            return self.X, self.y
        else:
            return self.X

    def read_data_features(self):
        """
        用于读取数据中的特征和标签
        :return:
        """
        self.X = np.array(self.df.loc[:, self.df.columns[8:]].fillna(value=0))  # 缺失值填充为0

    def read_data_sentence_vector(self):
        self.X = np.loadtxt(self.vec_path, delimiter=',')  # 缺失值填充为0

    def read_data_word_vector(self):
        """
        此方法暂时用不了
        :return:
        """
        for ind, file in enumerate(self.vec_path):
            if ind % 100 == 0:
                print("Dataset: {}".format(ind))
            data = np.loadtxt(file, delimiter=',')

            if self.X is not None:
                # print(self.X.shape, np.expand_dims(data, axis=0).shape)
                self.X = np.concatenate((self.X, np.expand_dims(data, axis=0)), axis=0)
            else:
                self.X = np.expand_dims(data, axis=0)

    def read_data_text(self):
        self.X = list(self.df.loc[:, "result"])

from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer

from read_data import *

writer = SummaryWriter("logs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_root = 'chinese_roberta_wwm_ext_pytorch'  # embedding或者句向量
config_path = "./{}/config.json".format(model_root)
model_path = "./{}/pytorch_model.bin".format(model_root)
vocab_path = "./{}/vocab.txt".format(model_root)
tokenizer = BertTokenizer.from_pretrained(vocab_path)
code_length = 32


class BertTextNet(nn.Module):
    def __init__(self, code_length):
        super(BertTextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained(config_path)
        self.textExtractor = BertModel.from_pretrained(model_path, config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)
        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features


def text_to_vector(text_data):
    tokens, segments, input_masks = [], [], []
    for text in text_data:
        text = "[CLS]" + text + "[SEP]"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    # max_len = max([len(single) for single in tokens])
    max_len = 144+2

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))  # 需要补0
        tokens[j] += padding  # input_ids
        input_masks[j] += padding  # input_mask
        segments[j] += padding  # segment_ids
        # segments列表全0，因为只有一个句子1，没有句子2
        # input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
        # 相当于告诉BertModel不要利用后面0的部分

    # 转换成PyTorch tensors
    tokens_tensor = torch.tensor(tokens)
    input_masks_tensors = torch.tensor(input_masks)
    segments_tensors = torch.tensor(segments)

    return tokens_tensor, input_masks_tensors, segments_tensors

    # # 提取文本特征
    # textNet = BertTextNet(code_length=code_length)  # 指定编码长度
    # text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
    # # print(text_hashCodes)
    # return text_hashCodes


class Datasets(data.Dataset):
    def __init__(self, x, y=None, ind=None):
        self.data = x
        self.label = y
        self.ind = ind
        if type(x[0]) is str:  # 说明要对文本进行编码
            self.input_ids, self.input_mask, self.segment_ids = text_to_vector(self.data)
        else:
            self.input_ids = None
            self.input_mask = None
            self.segment_ids = None

    def __getitem__(self, idx):
        if self.input_ids is not None:  # 说明是文本数据
            data = {"input_ids": self.input_ids[idx].to(device),  # input_ids 等同于 tokens_tensor
                    "input_mask": self.input_mask[idx].to(device),
                    "segment_ids": self.segment_ids[idx].to(device),
                    "input_x": self.data[idx],  # 文本
                    }
        else:
            data = {
                "input_ids": False,
                "input_mask": False,
                "segment_ids": False,
                "input_x": torch.tensor(self.data[idx]).to(device, dtype=torch.float),
            }
        if self.label is None:
            return data
        return data, torch.tensor(self.label[idx]).to(device), self.ind[idx]  # , dtype=torch.float

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    csv_path = "./all_data/features_all_data_0725.csv"
    vec_path = "./all_data/sentence_vectors_all.csv"
    data_x, data_y = ReadData(csv_path, vec_path).run(method="text")
    Datasets(data_x, data_y)

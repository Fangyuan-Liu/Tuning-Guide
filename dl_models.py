import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# embedding预训练权重：https://github.com/Embedding/Chinese-Word-Vectors/tree/master
# 加载方式：https://zhuanlan.zhihu.com/p/99362164
# 知识理解：https://www.cnblogs.com/jfdwd/p/11083750.html

class LSTM(nn.Module):
    def __init__(self, input_size=None, vocab_size=None, embedding_dim=None, cls=2, hidden_layer_size=100, num_layers=5):
        """
        https://blog.csdn.net/weixin_35757704/article/details/118389681
        https://blog.csdn.net/weixin_43646592/article/details/119192257
        LSTM二分类任务
        :param input_size: 输入数据的维度
        :param hidden_layer_size:隐层的数目
        :param cls: 分类数目
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        if embedding_dim is not None:
            self.word_embeddings = nn.Embedding(vocab_size,
                                                embedding_dim)  # embedding之后的shape: torch.Size([200, 8, 300])
            # self.word_embeddings.weight.data.copy_(torch.from_numpy(np.load("emb.npy")))
            self.word_embeddings.weight.requires_grad = True  # 微调or不微调
            self.word_embeddings_dim = embedding_dim
            self.lstm = nn.LSTM(embedding_dim, hidden_layer_size, batch_first=True, num_layers=num_layers)
            self.embedding = True
        else:
            self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, num_layers=num_layers)
            self.embedding = False
        self.linear = nn.Linear(hidden_layer_size, cls)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, input_mask, segment_ids, input_x):
        if self.embedding:
            inputs = self.word_embeddings(input_ids)
            # inputs = inputs.permute(0, 2, 1)
        else:
            inputs = input_x.view(len(input_x), 1, -1)  # .to(device, dtype=torch.float)
        hidden_cell = (torch.zeros(self.num_layers, inputs.shape[0], self.hidden_layer_size).to(device),  # shape: (n_layers, batch, hidden_size)
                       torch.zeros(self.num_layers, inputs.shape[0], self.hidden_layer_size).to(device))
        lstm_out, _ = self.lstm(inputs, hidden_cell)
        linear_out = self.linear(lstm_out[:, -1, :])  # =self.linear(lstm_out[:, -1, :])  self.linear(lstm_out.view(len(inputs), -1))
        predictions = self.sigmoid(linear_out)
        return predictions


class Bert(nn.Module):
    def __init__(self, cls, model_name="roberta", dropout_prob=0.1):  # cls: label的种类数
        super(Bert, self).__init__()
        if model_name == "bert":
            bert_dir = "./chinese_wwm_ext_pytorch"
        elif model_name == "roberta":
            bert_dir = "./chinese_roberta_wwm_ext_pytorch"
        else:
            raise "No this model_name !!!"
        config_path = bert_dir + "/config.json"

        assert os.path.exists(bert_dir) and os.path.exists(config_path), 'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir).to(device)
        self.bert_config = self.bert_module.config

        self.dropout_layer = nn.Dropout(dropout_prob).to(device)
        out_dims = self.bert_config.hidden_size
        self.obj_classifier = nn.Linear(out_dims, cls).to(device)

    def forward(self, input_ids, input_mask, segment_ids, input_x):
        bert_outputs = self.bert_module(
            input_ids=input_ids.to(device),
            attention_mask=input_mask.to(device),
            token_type_ids=segment_ids.to(device)
        )

        seq_out, pooled_out = bert_outputs[0], bert_outputs[1].to(device)
        # 对反向传播及逆行截断
        x = pooled_out.detach()
        out = self.obj_classifier(x)
        return out

""" 
以下是魔改Bert+TextCNN 
来源：https://zhuanlan.zhihu.com/p/422533717
"""
encode_layer=12
filter_sizes = [2, 2, 2]
num_filters = 3
n_class = 2
hidden_size = 768

class TextCNN_bert(nn.Module):
  def __init__(self):
    super(TextCNN_bert, self).__init__()
    self.num_filter_total = num_filters * len(filter_sizes)
    self.Weight = nn.Linear(self.num_filter_total, n_class, bias=False).to(device)
    self.bias = nn.Parameter(torch.ones([n_class]).to(device))
    self.filter_list = nn.ModuleList([
      nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
    ])

  def forward(self, input_x):
    # x: [bs, seq, hidden]
    x = input_x.unsqueeze(1) # [bs, channel=1, seq, hidden]

    pooled_outputs = []
    for i, conv in enumerate(self.filter_list.to(device)):
      h = F.relu(conv(x)) # [bs, channel=1, seq-kernel_size+1, 1]
      mp = nn.MaxPool2d(
        kernel_size = (encode_layer-filter_sizes[i]+1, 1)
      )
      # mp: [bs, channel=3, w, h]
      pooled = mp(h).permute(0, 3, 2, 1) # [bs, h=1, w=1, channel=3]
      pooled_outputs.append(pooled)

    h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [bs, h=1, w=1, channel=3 * 3]
    h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])

    output = self.Weight(h_pool_flat) + self.bias # [bs, n_class]


    return output

# model
class Bert_Blend_CNN(nn.Module):
    def __init__(self, cls=2):
        super(Bert_Blend_CNN, self).__init__()
        bert_dir = "./chinese_roberta_wwm_ext_pytorch"
        # self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.bert_module = BertModel.from_pretrained(bert_dir, output_hidden_states=True, return_dict=True).to(device)
        self.linear = nn.Linear(hidden_size, cls)
        self.textcnn = TextCNN_bert()

    def forward(self, input_ids, input_mask, segment_ids, input_x):
        bert_outputs = self.bert_module(
            input_ids=input_ids.to(device),
            attention_mask=input_mask.to(device),
            token_type_ids=segment_ids.to(device)
        )
        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]
        hidden_states = bert_outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1).to(device) # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
        for i in range(2, 13):
          cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1).to(device)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        logits = self.textcnn(cls_embeddings)
        return logits




class TextCNN(nn.Module):
    """
    代码来源：
    TextCNN的输入只能够是词向量吧，不能是已经提取好的特征
    https://blog.csdn.net/qsmx666/article/details/105302858
    https://blog.csdn.net/qq_37822951/article/details/90238522
    """
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels, cls=2):
        super(TextCNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding之后的shape: torch.Size([200, 8, 300])
        # self.word_embeddings.weight.data.copy_(torch.from_numpy(np.load("emb.npy")))
        self.word_embeddings.weight.requires_grad = True  # 微调or不微调
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), cls)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embedding_dim,
                                        out_channels = c,
                                        kernel_size = k))

    def forward(self, input_ids, input_mask, segment_ids, input_x):
        embeds = self.word_embeddings(input_ids)
        # embeds = self.word_embeddings(input_x)
        embeds = embeds.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结

        encoding = torch.cat([self.pool(F.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs



class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)


class TextCNN_1(nn.Module):
    """
    来源：https://blog.csdn.net/guyuealian/article/details/127846717
    """
    # vocab_size, embedding_dim, kernel_sizes, num_channels, cls=2
    def __init__(self, embedding_dim, kernel_sizes, num_channels, cls=2, vocab_size=-1, embeddings_pretrained=False):
        """
        :param num_classes: 输出维度(类别数num_classes)
        :param vocab_size: size of the dictionary of embeddings,词典的大小(vocab_size),
                               当vocab_size<0,模型会去除embedding层
        :param embedding_dim:  the size of each embedding vector，词向量特征长度
        :param kernel_sizes: CNN层卷积核大小
        :param num_channels: CNN层卷积核通道数
        :param embeddings_pretrained: embeddings pretrained参数，默认None
        :return:
        """
        super(TextCNN_1, self).__init__()
        self.num_classes = cls
        self.vocab_size = vocab_size
        # embedding层
        if self.vocab_size > 0:
            # embedding之后的shape: torch.Size([200, 8, 300])
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if embeddings_pretrained:
                # self.embedding = self.embedding.from_pretrained(embeddings_pretrained, freeze=False)
                self.embedding.weight.data.copy_(torch.from_numpy(np.load("emb.npy")))
        # 卷积层
        self.cnn_layers = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=c,
                          kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        # 最大池化层
        self.pool = GlobalMaxPool1d()
        # 输出层
        self.classify = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(sum(num_channels), self.num_classes)
        )

    def forward(self, input):
        """
        :param input:  (batch_size, context_size, embedding_size(in_channels))
        :return:
        """
        if self.vocab_size > 0:
            # 得到词嵌入(b,context_size)-->(b,context_size,embedding_dim)
            input = self.embedding(input)
            # (batch_size, context_size, channel)->(batch_size, channel, context_size)
        input = input.view(len(input), 1, -1).to(device, dtype=torch.float)
        input = input.permute(0, 2, 1)
        y = []
        for layer in self.cnn_layers:
            x = layer(input)
            x = self.pool(x).squeeze(-1)
            y.append(x)
        y = torch.cat(y, dim=1)
        out = self.classify(y)
        return out


if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 4
    num_classes = 2  # 输出类别
    context_size = 7  # 句子长度（字词个数）
    num_embeddings = 1024  # 词典的大小(vocab_size)
    embedding_dim = 6  # 词向量特征长度
    kernel_sizes = [2, 4]  # CNN层卷积核大小
    num_channels = [4, 5]  # CNN层卷积核通道数
    input = torch.ones(size=(batch_size, context_size)).long().to(device)
    model = TextCNN(num_classes=num_classes,
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    kernel_sizes=kernel_sizes,
                    num_channels=num_channels,
                    )
    model = model.to(device)
    model.eval()
    output = model(input)
    print("-----" * 10)
    print(model)
    print("-----" * 10)
    print(" input.shape:{}".format(input.shape))
    print("output.shape:{}".format(output.shape))
    print("-----" * 10)




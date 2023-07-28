import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from read_data import *
from dataset import *
from dl_models import *
from utils import *
from train_valid import *

import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.description = "please enter parameters......"
parser.add_argument("-batch", "--batch_size", help="this is the batch size", type=int, default="8")  # 128
parser.add_argument("-epoch", "--epochs", help="this is the epoch", type=int, default="10")
parser.add_argument("-lr", "--lr", help="this is the learning rate", type=float, default="0.001")
parser.add_argument("-model", "--model_name", help="this is the name of the model", type=str, default='roberta')
parser.add_argument("-method", "--word2vec", help="this is the way to transform the text data", type=str, default="text")
parser.add_argument("-norm", "--norm", help="whether normalization", type=bool, default="False")
args = parser.parse_args()

csv_path = "./all_data/features_all_data_0725.csv"
vec_path = "./all_data/sentence_vectors_all.csv"
model_dir = "./model"

batch_size = args.batch_size  # 64  # bert的只能跑16，Bert+TexcCNN只能跑8......
epochs = args.epochs  # 10
lr = args.lr  # 0.001
norm = args.norm

model_name = args.model_name  # "TextCNN"
word2vec = args.word2vec  # "text"

word2vec_list = ["feature", "sentence_vector", "text"]  # "word_vector"
model_name_list = ["bert", "roberta", "lstm_text", "lstm_feature", "TextCNN", "Bert_Blend_CNN"]

model_list = {
    "bert": "Bert(cls=2, model_name=model_name).to(device)",
    "roberta": "Bert(cls=2, model_name=model_name).to(device)",
    "lstm_text": "LSTM(vocab_size=21128, embedding_dim=300, cls=2).to(device)",  # input_size=ds['X_train'].shape[1],
    "lstm_feature": "LSTM(input_size=ds['X_train'].shape[1], cls=2).to(device)",
    "TextCNN": "TextCNN(vocab_size=21128, embedding_dim=300, kernel_sizes=[3, 4, 5], num_channels=[5, 5, 5]).to(device)",
    "Bert_Blend_CNN": "Bert_Blend_CNN(cls=2)"
}

random_state = 1111
setup_seed(1111)
data_x, data_y = ReadData(csv_path, vec_path).run(method=word2vec, norm=norm)

datasets_set = dataset_split_kfold(data_x, data_y)

y_predict_res = []
y_label_res = []
test_index_lst = []

for ind, ds in enumerate(datasets_set):  # 理论上来说，是1
    train_dataset = Datasets(ds['X_train'], ds['y_train'], ds["train_index"])
    val_dataset = Datasets(ds['X_test'], ds['y_test'], ds["test_index"])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print("Training loader prepared.")
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    print("Validation loader prepared.")

    model = eval(model_list[model_name])

    # loss_function = nn.BCELoss().to(device)  # 适合cls=1时使用
    loss_function = nn.CrossEntropyLoss().to(device)  # 适合cls=2时使用

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

    tv = TaV(model=model, epochs=epochs, optimizer=optimizer, loss_function=loss_function)

    best_acc = 0
    for i in range(epochs):
        print("-------第 {} 轮训练开始-------".format(i + 1))
        tv.train(train_loader)
        val_acc, out_lst, label_lst, ind_list = tv.valid(val_loader)
        if best_acc < val_acc:
            best_pred_lst = out_lst
            best_label_lst = label_lst
            best_ind_list = ind_list
            best_acc = val_acc

    y_label_res.extend(best_label_lst)
    y_predict_res.extend(best_pred_lst)
    test_index_lst.extend(best_ind_list)

df_res = pd.DataFrame({'test_index': test_index_lst,
                       'y_label': y_label_res,
                       'y_pred': y_predict_res})
df_res.to_excel("./output/{}_{}.xlsx".format(str(word2vec), str(model_name)), index=False)




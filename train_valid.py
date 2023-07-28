import torch
import torch.optim as optim
from dl_models import *

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TaV:
    def __init__(self, model, epochs, optimizer, loss_function):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, train_loader):
        # 开始训练
        self.model.train()
        train_loss = 0
        train_acc = 0
        output_list = []
        label_list = []
        for inputs, labels, _ in train_loader:
            self.optimizer.zero_grad()

            y_pred = self.model(**inputs).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            # y_pred_class = torch.round(y_pred)  # 如果cls是1的话，tensor([0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1., 1.],
            y_pred_class = y_pred.argmax(1)

            single_loss = self.loss_function(y_pred, labels)

            train_loss += single_loss.item()
            output_list += y_pred_class.cpu().detach().numpy().tolist()
            label_list += labels.cpu().detach().numpy().tolist()

            single_loss.backward()
            self.optimizer.step()

        train_acc = (np.array(output_list) - np.array(label_list)).tolist().count(0) / len(output_list)
        print("Train Loss: \t{}".format(train_loss))
        print("Train Acc: \t{}".format(train_acc))


    def valid(self, val_loader):
        # 开始验证
        self.model.eval()

        val_loss = 0
        val_acc = 0
        output_list = []
        label_list = []
        ind_list = []

        for inputs, labels, ind in val_loader:
            y_pred = self.model(**inputs).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            # y_pred_class = torch.round(y_pred)  # 如果cls是1的话，tensor([0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1., 1.],
            y_pred_class = y_pred.argmax(1)

            single_loss = self.loss_function(y_pred, labels)

            val_loss += single_loss.item()
            output_list += y_pred_class.cpu().detach().numpy().tolist()
            label_list += labels.cpu().detach().numpy().tolist()
            ind_list += ind.tolist()

        val_acc = (np.array(output_list) - np.array(label_list)).tolist().count(0) / len(output_list)
        print("Validation Loss: \t{}".format(val_loss))
        print("Validation Acc: \t{}".format(val_acc))
        return val_acc, output_list, label_list, ind_list

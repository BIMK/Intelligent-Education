import torch
import torch.nn as nn


class Net_loss(nn.Module):
    def __init__(self, num):
        self.input_size = num
        self.hidden_size = 8
        self.num_loss = 1

        super(Net_loss, self).__init__()
        self.sigmoid = nn.Sigmoid()  # 激活函数
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_loss)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x):
        out0 = self.dropout(self.fc1(x))
        out_loss = torch.sigmoid(self.fc2(out0))
        return out_loss


    def loss(self, output, label):
        loss = self.loss_function(output, label)
        return loss

    def load_snapshot(self, model, filename):
        f = open(filename, 'rb')
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close()

class Net_loss_ncd(nn.Module):
    def __init__(self, num):
        self.input_size = num
        self.hidden1_size, self.hidden2_size, self.hidden3_size = 64, 32, 8
        self.num_loss = 1

        super(Net_loss_ncd, self).__init__()
        self.sigmoid = nn.Sigmoid()  # 激活函数
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.input_size, self.hidden1_size)
        self.fc2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.fc3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.fc4 = nn.Linear(self.hidden3_size, self.num_loss)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x):
        out0 = self.dropout(self.fc1(x))
        out1 = self.dropout(self.fc2(out0))
        out2 = self.dropout(self.fc3(out1))
        out_loss = torch.sigmoid(self.fc4(out2))

        return out_loss

    def loss(self, output, label):
        loss = self.loss_function(output, label)
        return loss

    def load_snapshot(self, model, filename):
        f = open(filename, 'rb')
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close()

class Net_loss_ncd2(nn.Module):

    def __init__(self, num):
        self.input_size = num
        self.hidden1_size, self.hidden2_size = 16, 8
        self.num_loss = 1

        super(Net_loss_ncd2, self).__init__()
        self.sigmoid = nn.Sigmoid()  # 激活函数
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.input_size, self.hidden1_size)
        self.fc2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.fc3 = nn.Linear(self.hidden2_size, self.num_loss)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        out0 = self.dropout(self.fc1(x))
        out1 = (self.fc2(out0))
        out_loss = torch.sigmoid(self.fc3(out1))

        return out_loss


    def loss(self, output, label):
        loss = self.loss_function(output, label)
        return loss

    def load_snapshot(self, model, filename):
        f = open(filename, 'rb')
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close()














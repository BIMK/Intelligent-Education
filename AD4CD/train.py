import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from data_loader import exer_n, knowledge_n, student_n, MyDataSet, dataset, time_graph
from torch.utils.data import DataLoader
from model import Net, device
from util.pb4dl import pb4dl
from loss_function import MyLossFunction

epoch_n = 100

net = Net(student_n, exer_n, knowledge_n, time_graph)
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = MyLossFunction()

trainDataSet = MyDataSet("train")
validDataset = MyDataSet("valid")
num_workers = 4 if torch.cuda.is_available() else 0
trainDataLoader = DataLoader(trainDataSet, 256, pin_memory=True, num_workers=num_workers)
validDataLoader = DataLoader(validDataset, 256, pin_memory=True, num_workers=num_workers)


def train():
    net.train()
    running_loss = 0.0
    for i in pb4dl(trainDataLoader):
        studentId, problemId, skill_num, labels, time_taken, skill_index = i
        studentId, problemId, skill_num, labels, time_taken, skill_index = studentId.to(device), problemId.to(
            device), skill_num.to(
            device), labels.to(device), time_taken.to(device), skill_index.to(device)
        optimizer.zero_grad()
        output, recon_loss, mu, logvar = net.forward(studentId, problemId, skill_num, time_taken, skill_index)
        loss = loss_function(output, recon_loss, mu, logvar, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"train_loss={running_loss:.0f}")


def valid():
    net.eval()
    running_loss = 0.0
    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    with torch.no_grad():
        for i in validDataLoader:
            studentId, problemId, skill_num, labels, time_taken, skill_index = i
            studentId, problemId, skill_num, labels, time_taken, skill_index = studentId.to(device), problemId.to(
                device), skill_num.to(device), labels.to(device), time_taken.to(device), skill_index.to(device)
            output, recon_loss, mu, logvar = net.forward(studentId, problemId, skill_num, time_taken, skill_index)
            loss = loss_function(output, recon_loss, mu, logvar, labels)
            running_loss += loss.item()
            for a in range(len(labels)):
                if (labels[a] == 1 and output[a] > 0.5) or (labels[a] == 0 and output[a] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            pred_all += output.tolist()
            label_all += labels.tolist()

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        accuracy = correct_count / exer_count
        mse = mean_squared_error(label_all, pred_all)
        rmse = np.sqrt(mse)
        auc = roc_auc_score(label_all, pred_all)
        print(f"test_loss={running_loss:.0f}")
        print('accuracy= %f, mse= %f, rmse= %f, auc= %f' % (accuracy, mse, rmse, auc))
        with open('result/model_our.txt', 'a', encoding='utf8') as f:
            f.write('%s epoch= %d, accuracy= %f, mse= %f, rmse= %f, auc= %f\n' % (
                dataset, epoch + 1, accuracy, mse, rmse, auc))


if __name__ == '__main__':
    for epoch in range(epoch_n):
        train()
        valid()
        # torch.save(net, "model/" + f"/epoch{epoch + 1}.pth")

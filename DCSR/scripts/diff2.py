import logging
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
# from tensorflow import keras

from tqdm import trange
import json
import DiffModel2 as Diff
import pyat
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) 

cond = 'C++'
target = 'C'
batch = 256
epoch = 200

data_input = json.load(open(f'../datasets/PTADisc/{cond}/info.json', 'r'))
data_target = json.load(open(f'../datasets/PTADisc/{target}/info.json', 'r'))

input_dim = data_input['concept_cnt']
target_dim = data_target['concept_cnt']

dataset = target
# read datasets
test_triplets = pd.read_csv(f'../datasets/PTADisc/{dataset}/train_80_sort.csv', encoding='utf-8').to_records(
    index=False)
concept_map = json.load(open(f'../datasets/PTADisc/{dataset}/concept_map_{dataset}.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
dataInfo = json.load(open(f'../datasets/PTADisc/{dataset}/info.json', 'r'))

data = pyat.AdapTestDataset(test_triplets, concept_map,
                            dataInfo['train_cnt'], dataInfo['problem_cnt'], dataInfo['concept_cnt'])
print(dataInfo['problem_cnt'])

test_triplets_source = pd.read_csv(f'../datasets/PTADisc/{cond}/train_80_sort.csv', encoding='utf-8').to_records(
    index=False)
concept_map2 = json.load(open(f'../datasets/PTADisc/{cond}/concept_map_{cond}.json', 'r'))
concept_map2 = {int(k): v for k, v in concept_map2.items()}
dataInfo2 = json.load(open(f'../datasets/PTADisc/{cond}/info.json', 'r'))
data2 = pyat.AdapTestDataset(test_triplets_source, concept_map2,
                             dataInfo2['train_cnt'], dataInfo2['problem_cnt'], dataInfo2['concept_cnt'])
config = {
    'learning_rate': 0.0025,
    'batch_size': 2048,
    'num_epochs': 10,
    'num_dim': 1,
    'device': 'cpu',
}


class CAU_Model(nn.Module):
    def __init__(self, output_dim):
        super(CAU_Model, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(output_dim, output_dim)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.adjust_dim(x, self.output_dim)
        x = self.fc1(x)
        # x = self.ReLU(x)
        #x = self.fc2(x)
        return x

    def adjust_dim(self, data_cond1, output_dim):
      
        if data_cond1.shape[1] != output_dim:
          
            new_features = output_dim
            if data_cond1.shape[1] > new_features:
                data_cond1 = data_cond1[:, :new_features]
                # data_list = data_cond1.tolist()
                # for i in range(data_cond1.shape[1] - new_features):
                #     for j in range(len(data_list)):
                #         row = data_list[j]
                #         min_value = min(row, key=lambda x: abs(x))
                #         min_index = row.index(min_value)
                #         del row[min_index]
                # data_cond1 = torch.tensor(data_list)
            else:
                pad_size = new_features - data_cond1.shape[1]
                mean_values = torch.mean(data_cond1, dim=1, keepdim=True)
                data_cond1 = torch.nn.functional.pad(data_cond1, (0, pad_size))
        return data_cond1


class CustomDataset(Dataset):
    def __init__(self, input_file, target_file=None):
        with open(input_file, 'r') as f:
            self.input_data = json.load(f)

        if target_file is not None:
            with open(target_file, 'r') as f:
                self.target_data = json.load(f)
            assert len(self.input_data) == len(self.target_data), "Input and target data length must match"
        else:
            self.target_data = None

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.input_data[idx], dtype=torch.float32)
        if self.target_data is not None:
            target_tensor = torch.tensor(self.target_data[idx], dtype=torch.float32)
            return input_tensor, target_tensor
        else:
            return input_tensor


class CustomDataset_diff(Dataset):
    def __init__(self, data_tgt, data_cond, data_res):
        self.data_tgt = data_tgt
        self.data_cond = data_cond
        self.data_res = data_res

    def __len__(self):
        return len(self.data_tgt)

    def __getitem__(self, idx):
        return self.data_tgt[idx], self.data_cond[idx], self.data_res[idx]


class CustomDataset_diff_train(Dataset):
    def __init__(self, data_cond):
        self.data_cond = data_cond

    def __len__(self):
        return len(self.data_cond)

    def __getitem__(self, idx):
        return self.data_cond[idx]


def Get_diff_data():
    with open(f"../datasets/PTADisc/NCD+GMOCAT/{target}/{target}_stu_train_theta_unnum.json", "r") as file:
        data_tgt = json.load(file)
    with open(f"../datasets/PTADisc/NCD+GMOCAT/{cond}/{cond}_stu_train_theta_unnum.json", "r") as file:
        data_cond = json.load(file)
        # flat_list = [item for sublist in data_tgt for item in sublist]
        data_tgt = torch.tensor(data_tgt)
        # flat_list = [item for sublist in data_cond for item in sublist]
        data_cond = torch.tensor(data_cond)
    file_path = f'../datasets/PTADisc/{target}/train_80_sort_Diff.csv' 
    data_res = pd.read_csv(file_path)
    return data_tgt, data_cond, torch.tensor(data_res.values)


def train_Model(model, cond_file, target_file, IRT_Model, IRT_Model2):
    input_file = f'../datasets/PTADisc/NCD+GMOCAT/{cond_file}/{cond_file}_stu_train_theta_unnum.json'
    target_file = f'../datasets/PTADisc/NCD+GMOCAT/{target_file}/{target_file}_stu_train_theta_unnum.json'
    dataset_Model = CustomDataset(input_file, target_file)
    dataloader = DataLoader(dataset_Model, batch_size=batch, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01)


    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            predictions = model(inputs)
            loss1 = torch.abs(torch.norm(predictions - targets))
            # loss2 = CAU_loss(IRT_Model, IRT_Model2, predictions)
            # print("loss2:", loss2)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')


def test_Model(model, validation_file):
    validation_dataset = CustomDataset(validation_file)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in validation_dataloader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy().tolist())
    return torch.tensor(predictions)


def Diff_Model(diff_model, optimizer):
    print('=========Diff_Model========')
    IRT_Model = pyat.NCDModel(**config)
    IRT_Model.adaptest_init(data, target)
    IRT_Model.adaptest_preload(f'../models/ncd/{target}_80_train.pt')
    # evaluate models
    results = IRT_Model.adaptest_evaluate(data)
    # log results
    if results:
        for name, value in results.items():
            logging.info(f'{name}: {value}')
    else:
        logging.warning('No results returned from the evaluation.')

    IRT_Model2 = pyat.NCDModel(**config)
    IRT_Model2.adaptest_init(data2, cond)
    IRT_Model2.adaptest_preload(f'../models/ncd/{cond}_80_train.pt')
    # evaluate models
    results = IRT_Model2.adaptest_evaluate(data2)
    # log results
    if results:
        for name, value in results.items():
            logging.info(f'{name}: {value}')
    else:
        logging.warning('No results returned from the evaluation.')
    rcd_data = data
    data_tgt, data_cond, data_res = Get_diff_data()
    CAU_model = CAU_Model(target_dim)
    train_Model(CAU_model, cond, target, IRT_Model, IRT_Model2)
    val_File = f'../datasets/PTADisc/NCD+GMOCAT/{cond}/{cond}_stu_train_theta_unnum.json'
    Prediction = test_Model(CAU_model, val_File)

    diff_dataset = CustomDataset_diff(data_tgt, Prediction, data_res)
    data_loader = DataLoader(diff_dataset, batch_size=batch, shuffle=True)
    # Causalint_train(IRT_Model, IRT_Model2, CAU_model, data_cond, data_tgt, task=False)
    #
    # validation_file = f'../datasets/PTADisc/NCD+MAAT/Java/Java_stu_test_theta_unnum.json'
    # validation_dataset = CustomDataset(validation_file)
    # cond_data = Causalint_test(CAU_model, validation_dataset)
    # print(cond_data)

    for i in range(epoch):
        loss = Model_train(diff_model, optimizer, i, IRT_Model, data_loader, rcd_data)
        logging.info(f'loss:{loss}')
    eval_TrainSet(diff_model, CAU_model)
    rmse = eval_mae(diff_model, CAU_model)
    print('DIFF LOSS', loss.item(), 'RMSE: {}'.format(rmse))


def Model_train(diff_model, optimizer, epoch, IRT_Model, data_loader, rcd_data):

    print('Diff Training Epoch {}:'.format(epoch + 1))
    loss_ls = []
    batch_size = batch
    for data_tgt, data_cond, data_res in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{epoch}'):
        if data_cond.shape[1] != data_tgt.shape[1]:

            new_features = data_tgt.shape[1]
            if data_cond.shape[1] > new_features:
                data_cond = data_cond[:, :new_features]
                data_list = data_cond.tolist()
                for i in range(data_cond.shape[1] - new_features):
                    for j in range(len(data_list)):
                        row = data_list[j]
                        min_value = min(row, key=lambda x: abs(x))
                        min_index = row.index(min_value)
                        del row[min_index]
                data_cond = torch.tensor(data_list)
            else:
                pad_size = new_features - data_cond.shape[1]
                mean_values = int(torch.mean(torch.mean(data_cond, dim=1, keepdim=True)))
                data_cond = torch.nn.functional.pad(data_cond, (mean_values, pad_size))
        diff_model.train()
        # loss:
        is_task = False
        loss = train_loss(data_tgt, data_cond, diff_model, is_task, IRT_Model, data_res, rcd_data)
        diff_model.zero_grad()
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.)
        optimizer.step()

        # task loss
        is_task = True
        task_loss = train_loss(data_tgt, data_cond, diff_model, is_task, IRT_Model, data_res, rcd_data)
        diff_model.zero_grad()
        task_loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.)
        optimizer.step()
        loss_ls.append(task_loss.item())

    return torch.tensor(loss_ls).mean()


def Causalint_train(IRT_Model, IRT_Model2, CAU_Model, data_cond, data_tgt, task=False):
    print("---------Causality Train------------")
    optimizer = optim.Adam(CAU_Model.parameters(), lr=0.01)  
    input_file = f'../datasets/PTADisc/NCD+GMOCAT/Java/Java_stu_train_theta_unnum.json'
    target_file = f'../datasets/PTADisc/NCD+GMOCAT/DS/DS_stu_train_theta_unnum.json'
    dataset = CustomDataset(input_file, target_file)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    epochs = 100
   
    for epoch in range(epochs):
        CAU_Model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):

            predictions = CAU_Model(inputs)
            loss = torch.abs(torch.norm(targets - predictions))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}')


def Causalint_test(CAU_Model, data_cond):
    print("---------Causality Test-------------")
    dataset_val = data_cond
    CAU_Model.eval()

    validation_dataloader = DataLoader(dataset_val, batch_size=batch, shuffle=False)

    predictions = []
    with torch.no_grad():
        for inputs in tqdm(validation_dataloader):
            outputs = CAU_Model(inputs)
            predictions.extend(outputs.cpu().numpy())
    return torch.tensor(predictions)


def CAU_loss(IRT_Model, IRT_Model2, predictions):
    loss1 = 0
    loss2 = 0
    for i, pred_value in enumerate(predictions):
        user_data = test_triplets[test_triplets['user_id'] == i]
        item_id = user_data['item_id']
        score = user_data['score']

        for item, real in zip(item_id, score):
            data._knowledge_embs = data._knowledge_embs.to('cpu')
            knowledge_emb = data._knowledge_embs[item]
            pred_i = IRT_Model.forward_diff(pred_value, torch.LongTensor([item]), knowledge_emb)
            loss_i = torch.norm(real - pred_i)
            loss1 += loss_i
        loss1 = loss1 / len(item_id)

        user_data2 = test_triplets_source[test_triplets_source['user_id'] == i]
        item_id2 = user_data2['item_id']
        score2 = user_data2['score']

        if pred_value.shape[0] != input_dim:

            if pred_value.shape[0] > input_dim:
                pred_value = pred_value[:input_dim]
            else:
                pad_size = input_dim - pred_value.shape[0]
                pred_value = torch.nn.functional.pad(pred_value, (0, pad_size))

        for item2, real2 in zip(item_id2, score2):
            data2._knowledge_embs = data2._knowledge_embs.to('cpu')
            knowledge_emb2 = data2._knowledge_embs[item2]
            pred_i2 = IRT_Model2.forward_diff(pred_value, torch.LongTensor([item2]), knowledge_emb2)
            loss_i2 = torch.norm(real2 - pred_i2)
            loss2 += loss_i2
        loss2 = loss2 / len(item_id2)
    return loss1 - loss2


def train_loss(X, y, model, is_task, IRT_Model, data, rcd_data):
    tgt_emb = X
    cond_emb = y
    device = "cpu"
    loss = Diff.diffusion_loss_fn(model, tgt_emb, cond_emb, tgt_emb, device, is_task, IRT_Model, data, rcd_data)
    return loss


def eval_TrainSet(diff_model, CAU_model):
    print('Evaluating Train MAE:')
    predicts = list()
    batch_size = batch
    train_File = f'../datasets/PTADisc/NCD+GMOCAT/{cond}/{cond}_stu_train_theta_unnum.json'
    data_cond = test_Model(CAU_model, train_File)
    data_train_val = CustomDataset_diff_train(data_cond)
    data_loader = DataLoader(data_train_val, batch_size=batch_size, shuffle=False)
    model = diff_model
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(data_loader):
            outputs = test_model(inputs, model, 'cpu')
            predicts.extend(outputs.cpu().numpy().tolist())
    with open(f'../datasets/PTADisc/NCD+GMOCAT/Train/4_for_{target}/{cond}_for_{target}.json', 'w') as f:
        json.dump(predicts, f, indent=4)


def eval_mae(model, CAU_model):
    print('Evaluating MAE:')

    targets, predicts = list(), list()
    loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        with open(f"../datasets/PTADisc/NCD+GMOCAT/{cond}/{cond}_stu_test_theta_unnum.json", "r") as file:
            data_cond = json.load(file)
        with open(f"../datasets/PTADisc/NCD+GMOCAT/{target}/{target}_stu_test_theta_unnum.json", "r") as file:
            data_tgt = json.load(file)
        # flat_list = [item for sublist in data_tgt for item in sublist]
        data_tgt = torch.tensor(data_tgt)
        # # flat_list = [item for sublist in data_cond for item in sublist]
        data_cond = torch.tensor(data_cond)
        batch_size = batch
        val_File = f'../datasets/PTADisc/NCD+GMOCAT/{cond}/{cond}_stu_test_theta_unnum.json'
        data_cond = test_Model(CAU_model, val_File)
        data_val = CustomDataset_diff_train(data_cond)
        data_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
        # print(data_cond)
        for input in tqdm(data_loader):

            device = "cpu"
            pred = test_model(input, model, device)
            predicts.extend(pred.tolist())
            # print(targets)

        with open(f'../datasets/PTADisc/NCD+GMOCAT/4_for_{target}/{cond}_for_{target}.json', 'w') as f:
            json.dump(predicts, f, indent=4)
        return 0.1


def test_model(x, model, device):
    cond_emb = x
    pred_emb = Diff.p_sample_loop(model, cond_emb, device)
    # epsilon = 1e-12
    # logit_y = torch.log((trans_emb + epsilon) / (1 - trans_emb + epsilon))
    return pred_emb


def main():
    print("Diff_Model Start.")
    dataInfo = json.load(open(f'../datasets/PTADisc/{target}/info.json', 'r'))
    diff_steps = 1000
    diff_dim = batch
    diff_sample_steps = 30
    emb_dim = dataInfo['concept_cnt']
    diff_scale = 0.1
    diff_task_lambda = 0.1
    diff_mask_rate = 0.1
    diff_lr = 0.001
    diff_model = Diff.DiffCDR(diff_steps, diff_dim, emb_dim, diff_scale,
                              diff_sample_steps, diff_task_lambda, diff_mask_rate)
    optimizer_diff = torch.optim.Adam(params=diff_model.parameters(), lr=diff_lr)
    Diff_Model(diff_model, optimizer_diff)


if __name__ == "__main__":
    main()

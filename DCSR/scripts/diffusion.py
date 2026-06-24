import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
# from tensorflow import keras
import tqdm
import json
import DiffModel as Diff


def Diff_Model(diff_model, optimizer, model):
    print('=========Diff_Model========')
    for i in range(20):
        loss = Model_train(diff_model, optimizer, i, model)
    rmse = eval_mae(diff_model)
    print('DIFF LOSS', loss.item(), 'RMSE: {}'.format(rmse))


def Model_train(diff_model, optimizer, epoch, model):
    print('Training Epoch {}:'.format(epoch + 1))
    loss_ls = []
    with open(r"D:\IRT_fisher\datasets\PTADisc\C\C_train_stu_theta_80_unnum.json", "r") as file:
        data_tgt = json.load(file)
    with open(r"D:\IRT_fisher\datasets\PTADisc\DS\DS_train_stu_theta_80_unnum.json", "r") as file:
        data_cond = json.load(file)
        # flat_list = [item for sublist in data_tgt for item in sublist]
        data_tgt = torch.tensor(data_tgt)
        # flat_list = [item for sublist in data_cond for item in sublist]
        data_cond = torch.tensor(data_cond)
    file_path = r'D:\IRT_fisher\datasets\PTADisc\C\train_IRT_80_sort_Diff.csv' 
    data_res = pd.read_csv(file_path)
    batch_size = 8


    for i in tqdm.trange(0, len(data_tgt), batch_size, smoothing=0, mininterval=1.0):
        data_tgt1 = data_tgt[i:i + batch_size]
        data_cond1 = data_cond[i:i + batch_size]
        data_res1 = data_res[i:i + batch_size]

        diff_model.train()
        # loss:
        is_task = False
        loss = train_loss(data_tgt1, data_cond1, diff_model, is_task, model, data_res1)
        diff_model.zero_grad()
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.)
        optimizer.step()
        loss_ls.append(loss.item())

        # task loss
        is_task = True
        task_loss = train_loss(data_tgt1, data_cond1, diff_model, is_task, model, data_res1)
        diff_model.zero_grad()
        task_loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.)
        optimizer.step()

    return torch.tensor(loss_ls).mean()


def train_loss(X, y, model, is_task, IRT_model, data_log):
    tgt_emb = X
    cond_emb = y
    device = "cpu"
    loss = Diff.diffusion_loss_fn(model, tgt_emb, cond_emb, tgt_emb, device, is_task, IRT_model, data_log)
    return loss


def eval_mae(model):
    print('Evaluating MAE:')

    targets, predicts = list(), list()
    loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    with torch.no_grad():
        with open(r"D:\IRT_fisher\datasets\PTADisc\DS\DS_test_stu_theta_20_unnum.json", "r") as file:
            data_cond = json.load(file)
        with open(r"D:\IRT_fisher\datasets\PTADisc\C\C_test_stu_theta_20_unnum.json", "r") as file:
            data_tgt = json.load(file)
        # flat_list = [item for sublist in data_tgt for item in sublist]
        data_tgt = torch.tensor(data_tgt)
        # flat_list = [item for sublist in data_cond for item in sublist]
        data_cond = torch.tensor(data_cond)


        batch_size = 8

        for i in tqdm.trange(0, len(data_tgt), batch_size, smoothing=0, mininterval=1.0):
            data_tgt1 = data_tgt[i:i + batch_size]
            data_cond1 = data_cond[i:i + batch_size]
            device = "cpu"
            model.eval()
            pred = test_model(data_cond1, model, device)
            y_input = data_tgt1
            targets.extend(y_input.tolist())
            predicts.extend(pred.tolist())
            # print(targets)
        print(predicts)
        return torch.sqrt(mse_loss(targets, predicts)).item()


def test_model(x, model, device):
    cond_emb = x
    trans_emb = Diff.p_sample_loop(model, cond_emb, device)
    # epsilon = 1e-12
    # logit_y = torch.log((trans_emb + epsilon) / (1 - trans_emb + epsilon))
    return trans_emb


def main(model):
    print("Diff_Model Start.")
    diff_steps = 1000
    diff_dim = 8
    diff_sample_steps = 32
    emb_dim = 1
    diff_scale = 0.1
    diff_task_lambda = 0.1
    diff_mask_rate = 0.1
    diff_lr = 0.01
    diff_model = Diff.DiffCDR(diff_steps, diff_dim, emb_dim, diff_scale,
                              diff_sample_steps, diff_task_lambda, diff_mask_rate)
    optimizer_diff = torch.optim.Adam(params=diff_model.parameters(), lr=diff_lr)
    Diff_Model(diff_model, optimizer_diff, model)


# if __name__ == "__main__":
#     main()

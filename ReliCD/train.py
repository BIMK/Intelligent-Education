
from cmath import log
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from data_loader import TrainDataLoader, ValTestDataLoader
from correct_calibration import History
from predict import test
from model import Net
from utils import main
from metrics import get_metrics
import torch
import numpy as np
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter




def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    print('training model...')

    correctness_history = History(student_n, exer_n, knowledge_n)

    loss_batch = 0.0
    rmse_min = 1.0
    for epoch in range(epoch_n):
        net.train()


        data_loader.reset()
        running_loss = 0.0
        batch_count = 0

        while not data_loader.is_end():
            # with torch.autograd.set_detect_anomaly(True):

            loss_batch += 1

            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output, stu_mean, log_stu_covariance, kn_id, mean_mean = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, 'train')            
            loss = calculate_loss(output, mean_mean, stu_mean, log_stu_covariance, labels, input_stu_ids, kn_id, correctness_history, loss_batch)
                # print(loss)
                
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:

                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        rmse, _, ece = validate(net, epoch)
        # acc, rmse, auc = validate(net, epoch)
        # save_snapshot(net, 'model/model_epoch' + str(epoch + 1) )
        if rmse_min > rmse:
            rmse_min = rmse
            print('rmse_min=', rmse)
            save_snapshot(net, 'model/model_epoch')




def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()

    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    loss_function = nn.NLLLoss()
    running_loss = 0.0

    batch_count = 0
    pred_all, label_all = [], []
    pred_correct_incorrect = []

    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output, _, _ = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, 'val')

        output_0 = torch.ones(output.size()).to(device) - output
        output_1 = torch.cat((output_0, output), 1)        
        loss = loss_function(torch.log(output_1 + eps), labels)

        running_loss += loss.item()

        output = output.view(-1)

        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()
        pred_correct_incorrect += output_1.to(torch.device('cpu')).tolist()

    rmse, auc, ece = get_metrics(epoch, running_loss, batch_count, pred_all, label_all, pred_correct_incorrect, 'val')

    
    return rmse, auc, ece


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def calculate_loss(output_1, mean_mean, stu_mean, log_stu_covariance, labels, input_stu_ids, kn_id, correctness_history, loss_batch):
    loss_function = nn.NLLLoss()
    output_0 = torch.ones(output_1.size()).to(device) - output_1
    output = torch.cat((output_0, output_1), 1)
    loss_pred = loss_function(torch.log(output + eps), labels)

    a = torch.pow((stu_mean - mean_mean), 2) + log_stu_covariance.mul(2).exp() - log_stu_covariance.mul(2) - 1
    loss_kl =  0.5 * torch.mean(torch.sum(a, dim=1))


    # confidence is maximum class probability
    # 优化ranking loss
    stu_con_var = log_stu_covariance.mul(2).exp() * kn_id    # 获取学生在这道练习所对应的知识概念上的方差，kn_id进行掩码
    rank_1 = torch.sum(stu_con_var, dim = 1) / torch.count_nonzero(stu_con_var, dim = 1)
    rank_2 = torch.roll(rank_1, -1)

    stu_idx1 = input_stu_ids
    stu_idx2 = torch.roll(stu_idx1, -1)
    kno_idx1 = kn_id
    kno_idx2 = torch.roll(kno_idx1, -kno_idx1.shape[1])

    kno_index = torch.nonzero(kn_id)
    correct_concept = torch.zeros((input_stu_ids.shape[0], kn_id.shape[1]))

    pred_label = []
    for i in range(len(labels)):
        if (labels[i] == 1 and output_1[i] > 0.5) or (labels[i] == 0 and output_1[i] < 0.5):
            pred_label.append(1)
        else:
            pred_label.append(0)

    for i in (range(len(pred_label))):
            for j in range(len(kno_index)):
                if i == kno_index[j][0] and pred_label[i] == 1:
                    correct_concept[kno_index[j][0]][kno_index[j][1]] += 1
    

    rank_target, rank_margin = correctness_history.get_target_margin(stu_idx1, stu_idx2, kno_idx1, kno_idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_2 = rank_2 + (rank_margin / rank_target_nonzero)
    ranking_criterion = nn.MarginRankingLoss(margin=0.0)
    loss_ranking = ranking_criterion(rank_1, rank_2, rank_target)

    correctness_history.correctness_update(input_stu_ids, correct_concept)
    correctness_history.max_response_num_update(input_stu_ids, kn_id)
    
    loss = loss_pred + kl_weight * loss_kl + loss_ranking_weight*loss_ranking 



    return loss




if __name__ == '__main__':
    params = main()
    epoch_n = params.epoch
    device = params.device
    student_n = params.student_n
    exer_n = params.exer_n
    knowledge_n = params.knowledge_n
    lr = params.lr
    kl_weight = params.kl_weight
    loss_ranking_weight = params.loss_ranking_weight
    eps = params.eps


    train()

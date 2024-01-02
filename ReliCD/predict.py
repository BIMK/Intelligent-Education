import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from data_loader import ValTestDataLoader
from model import Net
from metrics import get_metrics
from vis_repre import vis_confidence
from utils import main
import torch.nn as nn



def test(epoch):
    data_loader = ValTestDataLoader('test')
    net = Net(student_n, exer_n, knowledge_n)
    # device = torch.device('cpu')
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, 'model/model_epoch')
    net = net.to(device)
    net.eval()

    loss_function = nn.NLLLoss()
    running_loss = 0.0
    batch_count = 0.0

    pred_all, label_all = [], []
    pred_correct_incorrect = []

    student_on_concept = []

    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)

        out_put, _, _ = net(input_stu_ids, input_exer_ids, input_knowledge_embs, 'test')
        output_0 = torch.ones(out_put.size()).to(device) - out_put
        output_1 = torch.cat((output_0, out_put), 1)        

        loss = loss_function(torch.log(output_1 + eps), labels)
        running_loss += loss.item()

        out_put = out_put.view(-1)

        pred_all += out_put.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()
        pred_correct_incorrect += output_1.to(torch.device('cpu')).tolist()
   
    rmse, auc, ece = get_metrics(epoch, running_loss, batch_count, pred_all, label_all, pred_correct_incorrect, 'test')
    # vis_confidence(pred_all, label_all)


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()



if __name__ == '__main__':
    params = main()
    device = params.device
    student_n = params.student_n
    exer_n = params.exer_n
    knowledge_n = params.knowledge_n
    eps = params.eps

    test(0)

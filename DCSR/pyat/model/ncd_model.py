import os
import time
import copy
import heapq
import json
import sys
import logging
from torch.nn.parameter import Parameter
from tqdm import tqdm
import torch

import torch.nn as nn
import numpy as np
import torch.utils.data as data
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
#from ..model.DNN_train import Net_loss, Net_loss_ncd, Net_loss_ncd2
from .abstract_model import AbstractModel
from ..utils import make_hot_vector
from ..utils.data import AdapTestDataset, TrainDataset, _Dataset
import pandas as pd
import matplotlib.pyplot as plt
import pyat
import math
import warnings
warnings.filterwarnings("ignore")

sys.path.append('../..')

def load_theta_from_json(stu_set):
    with open(f'../datasets/PTADisc/NCD+MAAT/Liner/{stu_set}/C++_for_{stu_set}.json', 'r') as f:
        embedding_data = json.load(f)
        embedding_tensor = torch.tensor(embedding_data)
    return embedding_tensor

class NCD(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable

        super(NCD, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NCDModel(AbstractModel, nn.Module):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None  # type: IRT

    @property
    def name(self):
        return 'ncd'

    def adaptest_init(self, data: _Dataset, stu_set=None):
        self.knowledge_dim = data.num_concepts
        self.exer_n = data.num_questions
        self.emb_num = data.num_students
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(NCDModel, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        if stu_set:
            self.student_emb = nn.Embedding.from_pretrained(load_theta_from_json(stu_set), freeze=False)
        print(self.student_emb.weight)

    def reinit(self):
        for name, param in self.named_parameters():
            if 'student_emb' in name:
                nn.init.xavier_normal_(param)

    def forward_diff(self, stu_emb, exer_id, kn_emb):
        '''
                :param stu_id: LongTensor
                :param exer_id: LongTensor
                :param kn_emb: FloatTensor, the knowledge relevancy vectors
                :return: FloatTensor, the probabilities of answering correctly
                '''
        # before prednet
        stu_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb.float()
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb.float()
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def forward_rule(self, stu_emb, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb.float()
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output


    def _apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def save_true(self, path):
        model_dict = self.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'student_emb' in k}
        torch.save(model_dict, path)

    def adaptest_train(self, train_data: TrainDataset, val_data):

        self.train()
        lr = self.config['learning_rate']
        bsz = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        logging.info('train on {}'.format(device))

        self.to(device)
        train_loader = data.DataLoader(train_data, batch_size=bsz, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 获取一开始初始化的theta值
        start_theta = self.student_emb.weight.clone()
        print('start_theta初始', start_theta)
        loss_function = nn.NLLLoss()
        rmse_b = 1.0
        best_epoch = 1
        for ep in range(1, epochs + 1):
            print('训练轮次：', ep)
            running_loss = 0.0
            batch_count = 0
            log_batch = 1
            pred_all, label_all = [], []
            exer_count, correct_count = 0, 0

            for student_ids, question_ids, correctness,a in train_loader:

                optimizer.zero_grad()
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                correctness = correctness.to(device).float()
                train_data._knowledge_embs = train_data._knowledge_embs.to(device)
                knowledge_embs = train_data._knowledge_embs[question_ids]
                pred_1 = self.forward(student_ids, question_ids, knowledge_embs)
                output = self.forward(student_ids, question_ids, knowledge_embs).view(-1)

                pred_all += pred_1.to(torch.device('cpu')).tolist()
                label_all += correctness.to(torch.device('cpu')).tolist()
                labels_np = correctness.cpu().numpy()
                output_np = output.cpu().detach().numpy()
                output_np[output_np >= 0.5] = 1
                output_np[output_np < 0.5] = 0
                eq_num_list = labels_np == output_np
                correct_count += eq_num_list.sum()
                exer_count += len(correctness)

                pred_0 = torch.ones(pred_1.size()).to(device) - pred_1
                pred = torch.cat((pred_0, pred_1), 1)
                loss = loss_function(torch.log(pred), correctness.long())
                loss.backward()
                optimizer.step()
                self._apply_clipper()
                batch_count += 1
                running_loss += loss.item()
                if batch_count % log_batch == 0:
                    # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, batch_count, running_loss / log_batch))
                    running_loss = 0.0
            pred_all = np.array(pred_all)
            label_all = np.array(label_all)
            # compute accuracy
            accuracy = correct_count / exer_count
            # compute AUC
            auc = roc_auc_score(label_all, pred_all)
           
            rmse1, auc1 = self.validate_ncd(train_data,  val_data)
           
            if rmse_b > rmse1:
                rmse_b = rmse1
                best_epoch = ep
                # self.adaptest_save(self.config['model_save_path'])
        print('best_epoch', best_epoch, 'min_rmse:', rmse_b)
        start_theta = self.student_emb.weight.clone()
        return start_theta

    def validate(self, train_data):
        dataset = 'assistment'
        device = 'cpu'
        # read datasets
        valid_triplets = pd.read_csv(f'../datasets/{dataset}/test_triplets.csv', encoding='utf-8').to_records(index=False)
        data = {}
        for sid, qid, correct in valid_triplets:
            data.setdefault(sid, {})
            data[sid].setdefault(qid, {})
            data[sid][qid] = correct
        real = []
        pred = []
        with torch.no_grad():
            self.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                knowledge_embs = train_data._knowledge_embs[question_ids]
                output = self.forward(student_ids, question_ids, knowledge_embs)
                output = output.view(-1)
                pred += output.tolist()
                real += [data[sid][qid] for qid in question_ids.cpu().numpy()]
            self.train()
        real = np.array(real)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        # compute RMSE
        rmse = np.sqrt(np.mean((real - pred) ** 2))
        return rmse, auc

    def validate_ncd(self, train_data, val_data):

        # read datasets
        data = val_data.data
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                knowledge_embs = train_data._knowledge_embs[question_ids]
                output = self.forward(student_ids, question_ids, knowledge_embs)
                output = output.view(-1)
                pred += output.tolist()
                real += [data[sid][qid] for qid in question_ids.cpu().numpy()]
            self.train()
        real = np.array(real)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        # compute RMSE
        rmse = np.sqrt(np.mean((real - pred) ** 2))
        return rmse, auc

    def adaptest_save(self, path):
        model_dict = self.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'student_emb' not in k}
        torch.save(model_dict, path)

   

    def gain_data(self, all_data, i):
        student_ids = []
        question_ids = []
        correctness = []
        for j in range(len(all_data)):
            student_ids.append(i)
        for k in all_data.keys():
            question_ids.append(k)
        for v in all_data.values():
            correctness.append(v)

        return student_ids,  question_ids,  correctness

    def divide(self, sid_list, qid_list, cor_list, rate):

        support_set_sid = sid_list[:int(len(qid_list) * rate)]
        query_set_sid = sid_list[int(len(qid_list) * rate):]
        support_set_qid = qid_list[:int(len(qid_list) * rate)]
        query_set_qid = qid_list[int(len(qid_list) * rate):]
        # update_set_qid = qid_list[int(len(qid_list) * rate *2):]
        support_set_cor = cor_list[:int(len(cor_list) * rate)]
        query_set_cor = cor_list[int(len(cor_list) * rate):]
        # update_set_cor = cor_list[int(len(cor_list) * rate * 2):]

        return support_set_sid, query_set_sid, support_set_qid, support_set_cor, query_set_qid, query_set_cor

    def adaptrain_preload(self, path, start_theta):
        self.student_emb.weight = Parameter(start_theta)
        self.load_state_dict(torch.load(path), strict=False)
        self.to(self.config['device'])

    def adaptest_preload(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        self.to(self.config['device'])

    def start_save(self, path):
        model_dict = self.state_dict()
        model_dict = {k: v for k, v in model_dict.items()}
        torch.save(model_dict, path)

    def save_snapshot(self, model, filename):
        f = open(filename, 'wb')
        torch.save(model.state_dict(), f)
        f.close()

    def adaptest_update(self, adaptest_data: AdapTestDataset):

        lr = self.config['learning_rate']
        bsz = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.NLLLoss()

        for name, param in self.named_parameters():
            if 'student_emb' not in name:
                param.requires_grad = False

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=bsz, shuffle=True)

        for ep in range(1, epochs + 1):
            running_loss = 0.0
            batch_count = 0
            log_batch = 100
            for student_ids, question_ids, correctness, a in dataloader:
                optimizer.zero_grad()
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                correctness = correctness.to(device).float()
                adaptest_data._knowledge_embs = adaptest_data._knowledge_embs.to(device)
                knowledge_embs = adaptest_data._knowledge_embs[question_ids]
                pred_1 = self.forward(student_ids, question_ids, knowledge_embs)
                pred_0 = torch.ones(pred_1.size()).to(device) - pred_1
                pred = torch.cat((pred_0, pred_1), 1)
                loss = loss_function(torch.log(pred), correctness.long())
                loss.backward()
                optimizer.step()
                self._apply_clipper()
                batch_count += 1
                running_loss += loss.item()
                # if batch_count % log_batch == 0:
                #     print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, batch_count, running_loss / log_batch))
                #     running_loss = 0.0
        return running_loss

    def adaptest_evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                adaptest_data._knowledge_embs = adaptest_data._knowledge_embs.to(device)
                knowledge_embs = adaptest_data._knowledge_embs[question_ids]
                output = self.forward(student_ids, question_ids, knowledge_embs)
                output = output.view(-1)
                pred += output.tolist()
                real += [data[sid][qid] for qid in question_ids.cpu().numpy()]
            self.train()

        c_count = 0
        for i in range(len(real)):
            if (real[i] == 1 and pred[i] >= 0.5) or (real[i] == 0 and pred[i] < 0.5):
                c_count += 1

        real = np.array(real)
        pred = np.array(pred)
        acc = c_count / len(real)
        auc = roc_auc_score(real, pred)

        return {
            'auc': auc,
            'acc': acc,
        }
        # coverages = []
        # for sid in data:
        #     all_concepts = set()
        #     tested_concepts = set()
        #     for qid in data[sid]:
        #         all_concepts |= set(concept_map[qid])
        #     for qid in adaptest_data.tested[sid]:
        #         tested_concepts |= set(concept_map[qid])
        #     coverage = len(tested_concepts) / len(all_concepts)
        #     coverages.append(coverage)
        # cov = sum(coverages) / len(coverages)
        #
        # real = np.array(real)
        # pred = np.array(pred)
        # auc = roc_auc_score(real, pred)
        #
        # return {
        #     'auc': auc,
        #     'cov': cov,
        # }


    def select_emc(self, sid, qid, kno_embs):
        theta = self.student_emb(torch.tensor(sid))
        stu_theta = torch.nn.Parameter(torch.tensor(theta, requires_grad=True, dtype=torch.float32))
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam([stu_theta], lr=lr)
        loss_function = nn.NLLLoss(reduce=False, size_average=False)

        num_items = len(qid)
        question_ids = torch.LongTensor(qid).to(device)
        batch_size = 32
        steps = int((num_items / batch_size)) + 1

        pos_grad_list = []
        neg_grad_list = []
        pred_list = []
        for step in range(steps):
            if step * batch_size == num_items:
                continue
            elif (step + 1) * batch_size >= num_items:  # 最后一次step中question可能不满足batchsize

                question_ids_tensor = question_ids[step * batch_size:]
            else:

                question_ids_tensor = question_ids[step * batch_size: (step + 1) * batch_size]
            user_theta = stu_theta.repeat(len(question_ids_tensor),1)
            theta_gf = stu_theta.repeat(len(question_ids_tensor))
            correct = torch.LongTensor([1]).repeat(len(question_ids_tensor)).to(device)
            wrong = torch.LongTensor([0]).repeat(len(question_ids_tensor)).to(device)
            kno_emb = kno_embs[question_ids_tensor]
            output_1 = self.forward_rule(user_theta, question_ids_tensor,kno_emb)
            preds = output_1.view(-1)
            pred_list.append(preds.cpu().detach().numpy())
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), correct)
            # loss = loss.reshape((len(loss),1))
            weight = torch.ones_like(loss)
            optimizer.zero_grad()
            pos_grad_batch = torch.autograd.grad(outputs=loss,
                                                 inputs=user_theta,
                                                 grad_outputs=weight,
                                                 retain_graph=True,
                                                 create_graph=True,
                                                 allow_unused=True,
                                                 only_inputs=True)
            pos_grad_batch = torch.norm(pos_grad_batch[0], dim=1)
            pos_grad_list.append(pos_grad_batch.cpu().detach().numpy())
            optimizer.zero_grad()
            loss = loss_function(torch.log(output), wrong)
            neg_grad_batch = torch.autograd.grad(outputs=loss,
                                                 inputs=user_theta,
                                                 grad_outputs=weight,
                                                 retain_graph=True,
                                                 create_graph=True,
                                                 allow_unused=True,
                                                 only_inputs=True)
            neg_grad_batch = torch.norm(neg_grad_batch[0], dim=1)
            neg_grad_list.append(neg_grad_batch.cpu().detach().numpy())

        pos_grad_list = np.concatenate(pos_grad_list, axis=0)
        neg_grad_list = np.concatenate(neg_grad_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        expected_change = pred_list * pos_grad_list + (1 - pred_list) * neg_grad_list
        expected_change = expected_change.tolist()

        return expected_change

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset):

        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        self.to(device)
        optimizer = torch.optim.Adam(self.student_emb.parameters(), lr=lr)
        loss_function = nn.NLLLoss()
        original_weights = self.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()
        adaptest_data._knowledge_embs = adaptest_data._knowledge_embs.to(device)

        knowledge_emb = adaptest_data._knowledge_embs[question_id]

        for ep in range(epochs):
            optimizer.zero_grad()
            pred_1 = self.forward(student_id, question_id, knowledge_emb)
            pred_0 = torch.ones(pred_1.size()).to(device) - pred_1
            pred = torch.cat((pred_0, pred_1), 1)
            loss = loss_function(torch.log(pred), correct.long())
            loss.backward()
            optimizer.step()
            self._apply_clipper()
        pos_weights = self.student_emb.weight.data.clone()
        self.student_emb.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred_1 = self.forward(student_id, question_id, knowledge_emb)
            pred_0 = torch.ones(pred_1.size()).to(device) - pred_1
            pred = torch.cat((pred_0, pred_1), 1)
            loss = loss_function(torch.log(pred), wrong.long())
            loss.backward()
            optimizer.step()
            self._apply_clipper()

        neg_weights = self.student_emb.weight.data.clone()
        self.student_emb.weight.data.copy_(original_weights)

        pred = self.forward(student_id, question_id,knowledge_emb).item()
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()

    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.forward(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.train()
        return pred_all

    def get_BE_weights(self, pred_all):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        d = 100
        Pre_true = {}
        Pre_false = {}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        w_ij_matrix = {}
        for i, _ in pred_all.items():
            w_ij_matrix[i] = {}
            for j, _ in pred_all.items():
                w_ij_matrix[i][j] = 0
        for i, _ in pred_all.items():
            for j, _ in pred_all.items():
                criterion_true_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 1)
                criterion_false_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 1)
                criterion_true_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 0)
                criterion_false_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 0)
                tensor_11 = torch.tensor(Pre_true[i], requires_grad=True)
                tensor_12 = torch.tensor(Pre_true[j], requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = Pre_false[i] * Pre_false[j] * diff_norm_00 + Pre_false[i] * Pre_true[j] * diff_norm_01 + \
                         Pre_true[i] * Pre_false[j] * diff_norm_10 + Pre_true[i] * Pre_true[j] * diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self, S_set, w_ij_matrix):
        res = 0.0
        for w_i in w_ij_matrix:
            if (w_i not in S_set):
                mx = float('-inf')
                for j in S_set:
                    if w_ij_matrix[w_i][j] > mx:
                        mx = w_ij_matrix[w_i][j]
                res += mx

        return res

    def delta_q_S_t(self, question_id, pred_all, S_set, sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            v: float, Each weight information
        """

        Sp_set = list(S_set)
        b_array = np.array(Sp_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
        sampled_dict = {key: value for key, value in pred_all.items() if key in sampled_elements}

        w_ij_matrix = self.get_BE_weights(sampled_dict)

        F_s = self.F_s_func(Sp_set, w_ij_matrix)

        Sp_set.append(question_id)
        F_sp = self.F_s_func(Sp_set, w_ij_matrix)
        return F_sp - F_s

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)




import os
import time
import copy
import math
import heapq
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt
from .abstract_model import AbstractModel
from ..utils import make_hot_vector
from ..utils.data import AdapTestDataset, TrainDataset, _Dataset
from ..model.DNN_train import Net_loss, Net_loss_ncd2

from tqdm import tqdm
import warnings
from torch.nn.parameter import Parameter
warnings.filterwarnings("ignore")



class IRT(nn.Module):
    def __init__(self, num_students, num_questions, num_dim):
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        self.alpha = nn.Embedding(self.num_questions, self.num_dim)
        self.beta = nn.Embedding(self.num_questions, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
        # pred = (alpha * theta) + beta
        pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = torch.sigmoid(pred)
        return pred

    def forward_update(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
        # pred = (alpha * theta) + beta
        pred = (alpha * theta).sum(dim=0, keepdim=True) + beta
        pred = torch.sigmoid(pred)
        return pred

    def forward_emc(self, theta, question_ids):
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
        if len(theta[0]) == 1:
            pred = (alpha * theta) + beta
        else:
            pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = torch.sigmoid(pred)
        return pred

    def get_knowledge_status(self, stu_ids):
        stu_emb = self.theta(stu_ids)
        return stu_emb.data


class IRTModel(AbstractModel):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None  # type: IRT

    @property
    def name(self):
        return 'Item Response Theory'

    def adaptest_init(self, data: _Dataset):
        self.model = IRT(data.num_students, data.num_questions, self.config['num_dim'])

    def adaptest_train(self, train_data: TrainDataset):
        lr = self.config['learning_rate']
        bsz = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        logging.info('train on {}'.format(device))

        self.model.to(device)
        train_loader = data.DataLoader(train_data, batch_size=bsz, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        start_theta = self.model.theta.weight.clone()

        for ep in range(1, epochs + 1):
            print('训练轮次：', ep)
            running_loss = 0.0
            batch_count = 0
            log_batch = 1
            for student_ids, question_ids, correctness in train_loader:
                optimizer.zero_grad()
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                correctness = correctness.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                loss = self._loss_function(pred, correctness)
                loss.backward()
                optimizer.step()
                batch_count += 1
                running_loss += loss.item()
                if batch_count % log_batch == 0:
                    # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, batch_count, running_loss / log_batch))
                    running_loss = 0.0

        self.start_save(self.config['model_after_param'])
        return start_theta

    def meta_pre(self, train_data: TrainDataset):
        lr = self.config['learning_rate']
        bsz = self.config['batch_size']
        device = self.config['device']
        logging.info('train on {}'.format(device))

        self.model.to(device)
        train_loader = data.DataLoader(train_data, batch_size=bsz, shuffle=True)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        meta_loss_list = []
        num = len(self.model.theta.weight[0]) + len(self.model.alpha.weight[0]) + len(self.model.beta.weight[0])
        loss_net = Net_loss(num)
        optimizer1 = torch.optim.Adam(loss_net.parameters(), lr=0.001)

        for ep in range(1, 6):
            meta_loss = 0
            user_id_list = list(train_loader.dataset.data.keys())
            for user_count in tqdm(range(len(user_id_list))):
                i = user_id_list[user_count]
                all_data = train_loader.dataset.data[i]
                student_ids, question_ids, correctness = self.gain_data(all_data, i)
                support_set_sid, query_set_sid, support_set_qid, support_set_cor, query_set_qid, query_set_cor = self.divide(student_ids, question_ids, correctness, 0.5)
                # Get the user's support_set and query_set
                theta = self.model.theta.weight[support_set_sid]
                alpha = self.model.alpha.weight[support_set_qid]
                beta = self.model.beta.weight[support_set_qid]

                for j in range(len(theta)):
                    if j == 0:
                        param = torch.cat((theta[j], alpha[j], beta[j]))
                        param = param.unsqueeze(0)
                    else:
                        add_param = torch.cat((theta[j], alpha[j], beta[j]))
                        add_param = add_param.unsqueeze(0)
                        param = torch.cat((param, add_param), dim=0)

                optimizer1.zero_grad()
                output_loss = loss_net.forward(param)
                output_loss1 = output_loss.clone()
                output_loss1 = output_loss1.tolist()
                max_index = list(map(output_loss1.index, heapq.nlargest(int(len(output_loss1)*0.4), output_loss1)))
                rmse_list = []
                for n in range(len(max_index)):
                    self.adaptest_preload(self.config['model_start_param'])
                    m_sid = support_set_sid[max_index[n]]
                    m_qid = support_set_qid[max_index[n]]
                    m_cor = support_set_cor[max_index[n]]

                    # Add the behavior record to update the theta parameter
                    optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)
                    for name, param in self.model.named_parameters():
                        if 'theta' not in name:
                            param.requires_grad = False

                    for epi in range(1, 100):
                        optimizer.zero_grad()
                        m_sid = torch.tensor(m_sid)
                        m_qid = torch.tensor(m_qid)
                        m_cor = torch.tensor(m_cor)
                        pred = self.model.forward_update(m_sid, m_qid).view(-1)
                        m_loss = self._loss_function(pred, m_cor)
                        m_loss.backward()
                        optimizer.step()

                    # theta is verified with query_set
                    with torch.no_grad():
                        qry_sid = torch.LongTensor(query_set_sid)
                        qry_qid = torch.LongTensor(query_set_qid)
                        # qry_cor = torch.LongTensor(query_set_cor)
                        output = self.model(qry_sid, qry_qid)
                        output = output.view(-1)
                        pred = np.array(output.tolist())
                        real = np.array(query_set_cor)
                        rmse = np.sqrt(np.mean(np.square(real - pred)))
                        rmse_list.append(rmse)

                max_loss = 0
                n = len(rmse_list)
                for p in range(len(rmse_list)-1):
                    for q in range(p+1, len(rmse_list)-p):
                        if rmse_list[p] < rmse_list[p+q]:
                            if output_loss1[max_index[p]] > output_loss1[max_index[p+q]]:
                                a = - (n-p)
                            else:
                                a = - (n-p)
                        else:
                            if output_loss1[max_index[p]] < output_loss1[max_index[p+q]]:
                                a = (n-p)
                            else:
                                a = (n-p)

                        max_loss = max_loss + max(0, a * (output_loss[max_index[p]] - output_loss[max_index[p+q]]))

                meta_loss += max_loss
                if max_loss != 0:
                    max_loss.backward()
                    optimizer1.step()

            ave_meta_loss = meta_loss.item() / len(train_loader)
            meta_loss_list.append(ave_meta_loss)
            print('meta_loss', ave_meta_loss)


        self.save_snapshot(loss_net, self.config['dnn_best_save_path'])
        plt.figure()
        plt.plot(range(len(meta_loss_list)), meta_loss_list, color='blue', linewidth=2.0, linestyle='--',
                 label='train_ave_loss')
        plt.legend()
        plt.title('ave-loss-figure')
        plt.show()

    def gain_rmse(self, max_index, support_set_sid, support_set_qid, support_set_cor, query_set_sid, query_set_qid, query_set_cor, rmse_list):
        lr = self.config['learning_rate']
        for n in max_index:
            self.adaptest_preload(self.config['model_start_param'])
            # print('start_theta', self.model.theta.weight[i])
            # max_index = output_loss.index(max(output_loss))
            m_sid = support_set_sid[n]
            m_qid = support_set_qid[n]
            m_cor = support_set_cor[n]

            # 加入行为记录进行theta参数更新
            optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)
            for name, param in self.model.named_parameters():
                if 'theta' not in name:
                    param.requires_grad = False

            for epi in range(1, 100):
                optimizer.zero_grad()
                m_sid = torch.tensor(m_sid)
                m_qid = torch.tensor(m_qid)
                m_cor = torch.tensor(m_cor)
                pred = self.model.forward_update(m_sid, m_qid).view(-1)
                m_loss = self._loss_function(pred, m_cor)
                # print('grad', theta.grad)
                # print('m_loss',m_loss)
                m_loss.backward()
                optimizer.step()

            # print(m_cor)
            # print('after_theta', self.model.theta.weight[i])
            # after_theta_list.append(self.model.theta.weight[i])
            # 用query_set来验证theta
            with torch.no_grad():
                qry_sid = torch.LongTensor(query_set_sid)
                qry_qid = torch.LongTensor(query_set_qid)
                # qry_cor = torch.LongTensor(query_set_cor)
                output = self.model(qry_sid, qry_qid)
                output = output.view(-1)
                pred = np.array(output.tolist())
                # print('pred', pred)
                real = np.array(query_set_cor)
                # print('real', real)
                rmse = np.sqrt(np.mean(np.square(real - pred)))
                # print('rmse', rmse)
                rmse_list.append(rmse)

        return rmse_list

    def update_theta(self, model, m_sid, m_qid, m_cor):
        # 加载模型初始化参数
        # print('start_theta', model.model.theta.weight.data[m_sid])
        lr = self.config['learning_rate']
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)
        for name, param in self.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        for epi in range(1, 50):
            optimizer.zero_grad()
            m_sid = torch.tensor(m_sid)
            m_qid = torch.tensor(m_qid)
            m_cor = torch.tensor(m_cor)
            pred = self.model.forward_update(m_sid, m_qid).view(-1)
            m_loss = self._loss_function(pred, m_cor)
            m_loss.backward()
            optimizer.step()

    def test_evaluate(self, sid, data):
        real = []
        pred = []
        with torch.no_grad():
            student_ids = [sid] * len(data[sid])
            question_ids = list(data[sid].keys())
            student_ids = torch.LongTensor(student_ids)
            question_ids = torch.LongTensor(question_ids)
            output = self.model(student_ids, question_ids)
            output = output.view(-1)
            pred += output.tolist()
            real += [data[sid][qid] for qid in question_ids.cpu().numpy()]
            pred = np.array(pred)
            real = np.array(real)
            rmse = np.sqrt(np.mean(np.square(real - pred)))

        return rmse

    def val_dnn(self,val_data, loss_net):
        lr = self.config['learning_rate']
        bsz = self.config['batch_size']
        device = self.config['device']
        logging.info('validate on {}'.format(device))
        val_loader = data.DataLoader(val_data, batch_size=bsz, shuffle=True)
        meta_val_list = []
        user_id_list = list(val_loader.dataset.data.keys())
        meta_loss = 0
        for i in user_id_list:
            # 获取每个用户的行为数据,验证集中有部分用户记录
            if len(val_loader.dataset.data[i]) < 4:
                continue
            all_data = val_loader.dataset.data[i]
            student_ids, question_ids, correctness = self.gain_data(all_data, i)
            # 划分数据集，按照5:5进行拆分成support_set: query_set
            support_set_sid, query_set_sid, support_set_qid, support_set_cor, query_set_qid, query_set_cor = self.divide(
                student_ids, question_ids, correctness, 0.5)
            # 获得用户的support_set 和 query_set
            theta = self.model.theta.weight[support_set_sid]
            alpha = self.model.alpha.weight[support_set_qid]
            beta = self.model.beta.weight[support_set_qid]

            for j in range(len(theta)):
                if j == 0:
                    param = torch.cat((theta[j], alpha[j], beta[j]))
                    param = param.unsqueeze(0)
                else:
                    add_param = torch.cat((theta[j], alpha[j], beta[j]))
                    # 升维
                    add_param = add_param.unsqueeze(0)
                    param = torch.cat((param, add_param), dim=0)

            # 经过DNN正向传播
            # loss_net = Net_loss(3)
            # optimizer1 = torch.optim.Adam(loss_net.parameters(), lr=0.001)
            # optimizer1.zero_grad()
            output_loss = loss_net.forward(param)
            output_loss1 = output_loss.clone()
            output_loss = output_loss.tolist()
            # 获取最大loss索引
            max_index = list(map(output_loss.index, heapq.nlargest(int(len(output_loss) * 1), output_loss)))
            rmse_list = []
            for n in range(len(max_index)):
                self.adaptest_preload(self.config['model_start_param'])
                # print('start_theta', self.model.theta.weight[0])
                # max_index = output_loss.index(max(output_loss))
                m_sid = support_set_sid[max_index[n]]
                m_qid = support_set_qid[max_index[n]]
                m_cor = support_set_cor[max_index[n]]

                # 加入行为记录进行theta参数更新
                optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)
                for name, param in self.model.named_parameters():
                    if 'theta' not in name:
                        param.requires_grad = False

                for epi in range(1, 100):
                    optimizer.zero_grad()
                    m_sid = torch.tensor(m_sid)
                    m_qid = torch.tensor(m_qid)
                    m_cor = torch.tensor(m_cor)
                    pred = self.model(m_sid, m_qid).view(-1)
                    m_loss = self._loss_function(pred, m_cor)
                    # print('grad', theta.grad)
                    # print('m_loss',m_loss)
                    m_loss.backward()
                    optimizer.step()

                # 用query_set来验证theta
                with torch.no_grad():
                    qry_sid = torch.LongTensor(query_set_sid)
                    qry_qid = torch.LongTensor(query_set_qid)
                    # qry_cor = torch.LongTensor(query_set_cor)
                    output = self.model(qry_sid, qry_qid)
                    output = output.view(-1)
                    pred = np.array(output.tolist())
                    # print('pred', pred)
                    real = np.array(query_set_cor)
                    # print('real', real)
                    rmse = np.sqrt(np.mean(np.square(real - pred)))
                    # print('rmse', rmse)
                    rmse_list.append(rmse)

            # print(rmse_list)

            # 设计偏序loss
            max_loss = 0
            n = len(rmse_list)
            for p in range(len(rmse_list) - 1):
                for q in range(p + 1, len(rmse_list) - p):
                    if rmse_list[p] < rmse_list[p + q]:
                        if output_loss1[max_index[p]] > output_loss1[max_index[p + q]]:
                            a = -(n - p)
                        else:
                            a = -(n - p)
                    else:
                        if output_loss1[max_index[p]] < output_loss1[max_index[p + q]]:
                            a = (n - p)
                        else:
                            a = (n - p)

                    max_loss = max_loss + max(0, a * (output_loss1[max_index[p]] - output_loss1[max_index[p + q]]))
            print(max_loss)
            meta_loss += max_loss


        ave_val_loss = meta_loss.item() / len(val_loader)
        meta_val_list.append(ave_val_loss)
        print('meta_loss', ave_val_loss)

        return ave_val_loss

    def adaptest_save(self, path):
        model_dict = self.model.state_dict()
        # save_alpha = model_dict['alpha.weight']
        # save_beta = model_dict['beta.weight']
        model_dict = {k: v for k, v in model_dict.items() if 'alpha' in k or 'beta' in k}
        torch.save(model_dict, path)

    def save_snapshot(self, model, filename):
        f = open(filename, 'wb')
        torch.save(model.state_dict(), f)
        f.close()

    def start_save(self, path):
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items()}
        torch.save(model_dict, path)

    def adaptest_preload(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptrain_preload(self, path, start_theta):
        self.model.theta.weight = Parameter(start_theta)
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])


    def adaptest_update(self, adaptest_data: AdapTestDataset):

        lr = self.config['learning_rate']
        bsz = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=bsz, shuffle=True)

        for ep in range(1, epochs + 1):
            running_loss = 0.0
            batch_count = 0
            log_batch = 100
            for student_ids, question_ids, correctness in dataloader:
                optimizer.zero_grad()
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                correctness = correctness.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                loss = self._loss_function(pred, correctness)
                loss.backward()
                optimizer.step()
                batch_count += 1
                running_loss += loss.item()
                if batch_count % log_batch == 0:
                    print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, batch_count, running_loss / log_batch))
                    running_loss = 0.0


    def adaptest_evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        # concept_map = adaptest_data.concept_map
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids)
                output = output.view(-1)
                pred += output.tolist()
                real += [data[sid][qid] for qid in question_ids.cpu().numpy()]
            self.model.train()

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


    def select_emc(self, sid, qid,kno_embs):
        theta = self.get_theta(sid)
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
            output_1 = self.model.forward_emc(user_theta, question_ids_tensor)
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = self.model.theta.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = self.model(student_id, question_id).item()
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()

    def irf(self, alpha, beta, theta):
        """ item response function
        """
        return 1.0 / (1.0 + np.exp(-alpha*(theta - beta)))

    def pd_irf_theta(self, alpha, beta, theta):
        """ partial derivative of item response function to theta

        :return:
        """
        p = IRTModel.irf(alpha, beta, theta)
        q = 1 - p
        return p * q * alpha

    def _loss_function(self, pred, real):
        return -(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()

    def get_alpha(self, question_id):
        return self.model.alpha.weight.data.numpy()[question_id]

    def get_beta(self, question_id):
        return self.model.beta.weight.data.numpy()[question_id]

    def get_theta(self, student_id):
        return self.model.theta.weight.data.numpy()[student_id]

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
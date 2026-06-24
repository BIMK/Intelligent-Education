import json
import sys
import random
import time
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score

from .abstract_model import AbstractModel
from ..utils import make_hot_vector
from ..utils.data import AdapTestDataset, TrainDataset, _Dataset

sys.path.append('..')
def load_theta_from_json(stu_set):
    print('load theta from json')
    with open(f'../datasets/PTADisc/IRT/{stu_set}/{stu_set}_stu_train_theta_unnum.json', 'r') as f:
        embedding_data = json.load(f)
        embedding_tensor = torch.tensor(embedding_data)
    return   embedding_tensor


class IRT(nn.Module):
    def __init__(self, num_students, num_questions, num_dim, stu_set=None):
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        # nn.Embedding
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        self.alpha = nn.Embedding(self.num_questions, self.num_dim)
        self.beta = nn.Embedding(self.num_questions, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        if stu_set:
            self.theta = nn.Embedding.from_pretrained(load_theta_from_json(stu_set), freeze=False)
        print(self.theta.weight)


    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
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
        self.model = IRT  # type: IRT

    @property
    def name(self):
        return 'Item Response Theory'

    def adaptest_init(self, data: _Dataset, stu_set=None): 
        self.model = IRT(data.num_students, data.num_questions, self.config['num_dim'], stu_set)

    def adaptest_train(self, train_data: TrainDataset):  
        lr = self.config['learning_rate'] 
        bsz = self.config['batch_size'] 
        epochs = self.config['num_epochs']  
        device = self.config['device'] 
        logging.info('train on {}'.format(device))

        self.model.to(device)
        train_loader = data.DataLoader(train_data, batch_size=bsz,
                                       shuffle=False) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  

        for ep in range(1, epochs + 1):
            running_loss = 0.0
            batch_count = 0
            log_batch = 1
            for student_ids, question_ids, correctness, _ in train_loader:
                # print(train_loader)
                # for student_ids, question_ids, correctness in train_loader:
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

        # student_ids = torch.LongTensor([0]).to(device)
        # theta_0 = self.model.theta(student_ids)
       
        # if theta_0.item() < 0:  
        #     with torch.no_grad():
        #         self.model.theta.weight.data = -self.model.theta.weight.data 
        #         self.model.alpha.weight.data = -self.model.alpha.weight.data  
        #    

    def adaptest_save(self, path): 
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'alpha' in k or 'beta' in k}
        torch.save(model_dict, path)

    def adaptest_preload(self, path): 
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self, adaptest_data: AdapTestDataset): 

        lr = self.config['learning_rate']
        bsz = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)

        # for name, param in self.model.named_parameters():
        #     if 'theta' not in name:
        #         param.requires_grad = False

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=bsz, shuffle=True)

        for ep in range(1, epochs + 1):
            running_loss = 0.0
            batch_count = 0
            log_batch = 100
            for student_ids, question_ids, correctness, _ in dataloader:
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

        print(self.get_theta(1))
        # for name, param in self.model.named_parameters():
        #     param.requires_grad = True

    def adaptest_evaluate(self, adaptest_data: AdapTestDataset): 
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
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

        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts |= set(concept_map[qid])
            for qid in adaptest_data.tested[sid]:
                tested_concepts |= set(concept_map[qid])
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages)  
        # Rate，FPR）

        real = np.array(real)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)  

        
        pred_labels = (pred >= 0.5).astype(int)

        
        acc = accuracy_score(real, pred_labels)

        return {
            'auc': auc,
            'acc': acc,
            'cov': cov,
        }
   
    # Quality Module
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

   
    def fisher_information(self, model, alpha, beta, theta):
        """ calculate the fisher information
        """
        try:
            information = []
            for t in theta:
                p = self.irf(alpha, beta, t)
                q = 1 - p
                pdt = self.pd_irf_theta(alpha, beta, t)
                # information.append((pdt**2) / (p * q))
                information.append(p * q * (alpha ** 2))
            information = np.array(information)
            return information
        except TypeError:
            p = self.irf(alpha, beta, theta)
            q = 1 - p
            # pdt = model.pd_irf_theta(alpha, beta, theta)
            # return (pdt ** 2) / (p * q + 1e-7)
            return (p * q * (alpha ** 2))

    def irf(self, alpha, beta, theta):
        """ item response function
        """
        return 1.0 / (1.0 + np.exp(-alpha * (theta - beta)))

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

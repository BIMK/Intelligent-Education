import torch.nn as nn
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class LSTMKT(nn.Module):
    def __init__(self, params):
        super(LSTMKT, self).__init__()
        self.hidden_size = params.hidden_size
        self.embed_size = params.embed_size
        self.problem_num = params.problem_num
        self.skill_num = params.skill_num
        self.num_layer = 1
        self.device = params.device
        self.problem_emb = nn.Embedding(self.problem_num + 1, self.embed_size)
        self.inter_embedding = torch.nn.Embedding(self.skill_num * 2 + 1, self.embed_size,
                                                  padding_idx=self.skill_num)
        self.L1 = nn.Linear(self.embed_size * 2, self.embed_size)
        self.rnn = torch.nn.LSTM(input_size=self.embed_size,
                                 hidden_size=self.hidden_size,
                                 batch_first=True,
                                 num_layers=self.num_layer)
        self.args = params
        self.out = nn.Sequential(
            torch.nn.Linear(self.args.hidden_size + self.args.embed_size, self.args.embed_size),
            torch.nn.Linear(self.args.embed_size, 1),
            nn.Dropout(p=0.25)
        )
        self.loss_function = torch.nn.BCELoss()

    def forward(self, log_dict):
        seqs_length = log_dict['seqs_length'].long()
        problem_seq_tensor = log_dict['problem_seqs_tensor'].to(self.device)
        skill_seqs_tensor = log_dict['skill_seqs_tensor'].to(self.device)
        correct_seqs_tensor = log_dict['correct_seqs_tensor'].to(self.device)
        mask_labels = correct_seqs_tensor * (correct_seqs_tensor > -1).long().to(self.device) 
        inter_embed_tensor = self.inter_embedding(skill_seqs_tensor + self.skill_num * mask_labels)  
        problem_emb = self.problem_emb(problem_seq_tensor)
        inter_embed_tensor = self.L1(torch.cat([problem_emb, inter_embed_tensor], dim=-1))
        output, _ = self.rnn(inter_embed_tensor)
        prediction_seq_tensor = self.out(torch.cat([output, problem_emb], dim=-1))
        target = correct_seqs_tensor[:, 1:]
        labels = target.reshape(-1)
        m = torch.nn.Sigmoid()
        preds = (prediction_seq_tensor[:, :-1].reshape(-1))
        mask = labels > -1
        masked_labels = labels[mask].float()
        masked_preds = m(preds[mask])
        out_dict = {'predictions': masked_preds, 'labels': masked_labels}
        return out_dict

    def getPreds(self, h, problem):
        with torch.no_grad():
            problem = self.problem_emb(problem)
            preds = self.out(torch.cat([h, problem], dim=-1))
            m = torch.nn.Sigmoid()
            preds = m(preds)
        return preds

    def loss(self, outdict):
        predictions = outdict['predictions']
        labels = outdict['labels']
        labels = torch.as_tensor(labels, dtype=torch.float)
        loss = self.loss_function(predictions, labels)
        return loss

    def getks(self, log_dict):
        with torch.no_grad():
            problem_seq_tensor = log_dict['problem_seqs_tensor'].to(self.device)
            skill_seqs_tensor = log_dict['skill_seqs_tensor'].to(self.device)
            correct_seqs_tensor = log_dict['correct_seqs_tensor'].to(self.device)
            mask_labels = correct_seqs_tensor * (correct_seqs_tensor > -1).long().to(self.device)  
            inter_embed_tensor = self.inter_embedding(skill_seqs_tensor + self.skill_num * mask_labels) 
            problem_emb = self.problem_emb(problem_seq_tensor)
            inter_embed_tensor = self.L1(torch.cat([problem_emb, inter_embed_tensor], dim=-1))
            output, _ = self.rnn(inter_embed_tensor)
        return output

    def getskill(self, skill):
        with torch.no_grad():
            skill = self.problem_emb(skill)
        return skill

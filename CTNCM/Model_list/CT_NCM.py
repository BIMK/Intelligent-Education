# -*-coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class CT_NCM(nn.Module):
    def __init__(self, dataset, skill_num, problem_num, device,
                 hidden_size, embed_size, prelen1, prelen2, dropout1, dropout2):

        super(CT_NCM, self).__init__()
        self.device = device

        self.skill_num = skill_num
        self.problem_num = problem_num

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.knowledge_dim = self.hidden_size
        self.input_len = self.knowledge_dim
        self.prelen1 = prelen1
        self.prelen2 = prelen2
        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)

        self.inter_embedding = torch.nn.Embedding(2 * self.skill_num, self.embed_size)
        self.reclstm = torch.nn.Linear(self.embed_size + self.hidden_size, 7 * self.hidden_size)

        self.problem_disc = torch.nn.Embedding(self.problem_num, 1)
        self.problem_diff = torch.nn.Embedding(self.problem_num, self.knowledge_dim)

        self.linear1 = torch.nn.Linear(self.input_len, self.prelen1)
        self.linear2 = torch.nn.Linear(self.prelen1, self.prelen2)
        self.linear3 = torch.nn.Linear(self.prelen2, 1)

        self.loss_function = torch.nn.BCELoss()

        if dataset in {'Junyi', 'Ednet'}:
            for name, param in self.named_parameters():
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, log_dict):
        problem_seqs_tensor = log_dict['problem_seqs_tensor'][:,1:].to(self.device)
        skill_seqs_tensor = log_dict['skill_seqs_tensor'].to(self.device)
        time_lag_seqs_tensor = log_dict['time_lag_seqs_tensor'][:,1:].to(self.device)
        correct_seqs_tensor = log_dict['correct_seqs_tensor'].to(self.device)

        seqs_length = log_dict['seqs_length'].to(self.device)
        mask_labels = correct_seqs_tensor * (correct_seqs_tensor > -1).long()

        inter_embed_tensor = self.inter_embedding(skill_seqs_tensor + self.skill_num * mask_labels)
        batch_size = correct_seqs_tensor.size()[0]
        hidden, _ = self.continues_lstm(inter_embed_tensor, time_lag_seqs_tensor, seqs_length, batch_size)

        hidden_packed = torch.nn.utils.rnn.pack_padded_sequence(hidden[1:,],
                                                                seqs_length.cpu()-1,
                                                                batch_first=False)
        theta = hidden_packed.data
        problem_packed = torch.nn.utils.rnn.pack_padded_sequence(problem_seqs_tensor,
                                                                 seqs_length.cpu()-1,
                                                                 batch_first=True)
        predictions = torch.squeeze(self.problem_hidden(theta, problem_packed.data))
        labels_packed = torch.nn.utils.rnn.pack_padded_sequence(correct_seqs_tensor[:,1:],
                                                                seqs_length.cpu()-1,
                                                                batch_first=True)
        labels = labels_packed.data
        out_dict = {'predictions': predictions, 'labels': labels}
        return out_dict

    def continues_lstm(self, inter_embed_tensor, time_lag_seqs_tensor, seqs_length, batch_size):
        self.init_states(batch_size=batch_size)
        h_list = []
        h_list.append(self.h_delay)
        for t in range(max(seqs_length) - 1):
            one_batch = inter_embed_tensor[:, t]
            c, self.c_bar, output_t, delay_t = \
                self.conti_lstm(one_batch, self.h_delay, self.c_delay,
                                self.c_bar)
            time_lag_batch = time_lag_seqs_tensor[:, t]
            self.c_delay, self.h_delay = \
                self.delay(c, self.c_bar, output_t, delay_t, time_lag_batch)
            self.h_delay = torch.as_tensor(self.h_delay, dtype=torch.float)
            h_list.append(self.h_delay)
        hidden = torch.stack(h_list)

        return hidden, seqs_length

    def init_states(self, batch_size):
        self.h_delay = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c_delay = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c_bar = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)

    def conti_lstm(self, one_batch_inter_embed, h_d_t, c_d_t, c_bar_t):
        input = torch.cat((one_batch_inter_embed, h_d_t), dim=1)
        (i, f, z, o, i_bar, f_bar, delay) = torch.chunk(self.reclstm(input), 7, -1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        z = torch.tanh(z)
        o = torch.sigmoid(o)
        i_bar = torch.sigmoid(i_bar)
        f_bar = torch.sigmoid(f_bar)
        delay = F.softplus(delay)
        c_t = f * c_d_t + i * z
        c_bar_t = f_bar * c_bar_t + i_bar * z
        return c_t, c_bar_t, o, delay

    def delay(self, c, c_bar, output, delay, time_lag):
        c_delay = c_bar + (c - c_bar) * torch.exp(- delay * time_lag.unsqueeze(-1))
        h_delay = output * torch.tanh(c_delay)
        return c_delay, h_delay

    def problem_hidden(self, theta, problem_data):
        problem_diff = torch.sigmoid(self.problem_diff(problem_data))
        problem_disc = torch.sigmoid(self.problem_disc(problem_data))
        input_x = (theta - problem_diff) * problem_disc * 10
        input_x = self.dropout1(torch.sigmoid(self.linear1(input_x)))
        input_x = self.dropout2(torch.sigmoid(self.linear2(input_x)))
        output = torch.sigmoid(self.linear3(input_x))
        return output

    def loss(self, outdict):
        predictions = outdict['predictions']
        labels = outdict['labels']
        labels = torch.as_tensor(labels, dtype=torch.float)
        loss = self.loss_function(predictions, labels)
        return loss

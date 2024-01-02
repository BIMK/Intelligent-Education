# -*-coding:utf-8 -*-

import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from Datareader import Datareader
from tqdm import tqdm


class CTLSTMDataset(Dataset):

    def __init__(self,
                 dataset='Assistment12',
                 mode = 'train',
                 max_length=100,
                 cross_validation=False,
                 k_fold_num=0):

        assert mode == 'train' or mode == 'test' or mode == 'val'

        if mode == 'train':
            filename = 'TrainSet'
        elif mode == 'val':
            filename = 'ValSet'
        else:
            filename = 'TestSet'
        self.file_path = 'data/' + dataset + '/' + str(filename) + '/' + str(mode) + str(k_fold_num) + '_100.csv'
        data_use = pd.read_csv(self.file_path)
        self.len = len(data_use)

        self.time_lag_seq = data_use['time_lag']
        self.timestamp = data_use['timestamp']
        self.problem_seq = data_use['problem_seq']
        self.skill_seq = data_use['skill_seq']
        self.correct_seq = data_use['correct_seq']

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        batch_dict = {
            'time_lag_seq': list(map(float, self.time_lag_seq[item].strip('[').strip(']').split(','))),
            'timestamp_seq': list(map(float, self.timestamp[item].strip('[').strip(']').split(','))),
            'problem_seq': list(map(int, self.problem_seq[item].strip('[').strip(']').split(','))),
            'skill_seq': list(map(int, self.skill_seq[item].strip('[').strip(']').split(','))),
            'correct_seq': list(map(eval, self.correct_seq[item].strip('[').strip(']').split(',')))
        }

        return batch_dict

# Use this method to compose a batch from multiple samples
def pad_batch_fn(many_batch_dict):
    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['problem_seq']), reverse=True)
    problem_seqs = [torch.LongTensor(seq['problem_seq']) for seq in sorted_batch]
    skill_seqs = [torch.LongTensor(seq['skill_seq']) for seq in sorted_batch]
    time_lag_seqs = [torch.FloatTensor(seq['time_lag_seq']) for seq in sorted_batch]
    correct_seqs = [torch.LongTensor(seq['correct_seq']) for seq in sorted_batch]
    timestamp_seqs = [torch.FloatTensor(seq['timestamp_seq']) for seq in sorted_batch]

    seqs_length = torch.LongTensor(list(map(len, skill_seqs)))

    problem_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    skill_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    time_lag_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
    correct_seqs_tensor = torch.full((len(sorted_batch), seqs_length.max()), -1).long()
    timestamp_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()

    for idx, (problem_seq, skill_seq, time_lag_seq, correct_seq, timestamp_seq, seq_len) in enumerate(zip(problem_seqs, skill_seqs, time_lag_seqs, correct_seqs, timestamp_seqs, seqs_length)):
        problem_seqs_tensor[idx, :seq_len] = torch.LongTensor(problem_seq)
        skill_seqs_tensor[idx, :seq_len] = torch.LongTensor(skill_seq)
        time_lag_seqs_tensor[idx, :seq_len] = torch.FloatTensor(time_lag_seq)
        correct_seqs_tensor[idx, :seq_len] = torch.LongTensor(correct_seq)
        timestamp_seqs_tensor[idx, :seq_len] = torch.FloatTensor(timestamp_seq)

    return_dict = {'problem_seqs_tensor': problem_seqs_tensor,
                   'skill_seqs_tensor': skill_seqs_tensor,
                   'time_lag_seqs_tensor': time_lag_seqs_tensor,
                   'correct_seqs_tensor': correct_seqs_tensor,
                   'timestamp_seqs_tensor': timestamp_seqs_tensor,
                   'seqs_length': seqs_length}
    return return_dict










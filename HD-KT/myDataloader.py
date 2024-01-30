import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Datareader import Datareader
from tqdm import tqdm


class CTLSTMDataset(Dataset):

    def __init__(self,
                 dataset='Assistment',
                 mode='train',
                 max_length=100,
                 cross_validation=False,
                 k_fold_num=0):

        assert mode == 'train' or mode == 'test' or mode == 'val'

        if cross_validation == False:
            self.file_path = 'data' + '/' + dataset + '/' + 'data_information_' + str(max_length) + '.pkl'

        else:
            self.file_path = 'data/' + dataset + '/data_information_' + str(max_length) + '_' + str(k_fold_num) + '.pkl'

        pkl_file = open(self.file_path, 'rb')
        data=pd.read_pickle(pkl_file)
        pkl_file.close()

        data_use = data.data_df[mode]
        self.len = len(data_use)
        self.user_id = data_use['user_id']
        self.time_lag_seq = data_use['time_lag']
        self.timestamp = data_use['timestamp']
        self.problem_seq = data_use['problem_seq']
        self.skill_seq = data_use['skill_seq']
        self.correct_seq = data_use['correct_seq']

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        batch_dict = {
            'user_id': self.user_id[item],
            'time_lag_seq': self.time_lag_seq[item],
            'timestamp_seq': self.timestamp[item],
            'problem_seq': self.problem_seq[item],
            'skill_seq': self.skill_seq[item],
            'correct_seq': self.correct_seq[item]
        }

        return batch_dict



def pad_batch_fn(many_batch_dict):

    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['problem_seq']), reverse=True)

    problem_seqs = [torch.LongTensor(seq['problem_seq']) for seq in sorted_batch]
    skill_seqs = [torch.LongTensor(seq['skill_seq']) for seq in sorted_batch]
    time_lag_seqs = [torch.FloatTensor(seq['time_lag_seq']) for seq in sorted_batch]
    correct_seqs = [torch.LongTensor(seq['correct_seq']) for seq in sorted_batch]
    timestamp_seqs = [torch.FloatTensor(seq['timestamp_seq']) for seq in sorted_batch]

    seqs_length = torch.LongTensor(list(map(len, skill_seqs)))
    user_ids = [seq['user_id'] for seq in sorted_batch]

    user_ids_tensor = torch.zeros(len(sorted_batch), ).long()
    problem_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    skill_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    time_lag_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
    correct_seqs_tensor = torch.full((len(sorted_batch), seqs_length.max()), -1).long()
    timestamp_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()


    for idx, (user_id,problem_seq, skill_seq, time_lag_seq, correct_seq, timestamp_seq, seq_len) in enumerate(
            zip(user_ids,problem_seqs, skill_seqs, time_lag_seqs, correct_seqs, timestamp_seqs, seqs_length)):
        user_ids_tensor[idx] = user_id
        problem_seqs_tensor[idx, :seq_len] = torch.LongTensor(problem_seq)
        skill_seqs_tensor[idx, :seq_len] = torch.LongTensor(skill_seq)
        time_lag_seqs_tensor[idx, :seq_len] = torch.FloatTensor(time_lag_seq)
        correct_seqs_tensor[idx, :seq_len] = torch.LongTensor(correct_seq)
        timestamp_seqs_tensor[idx, :seq_len] = torch.FloatTensor(timestamp_seq)

    return_dict = {'user_id': user_ids_tensor,
                    'problem_seqs_tensor': problem_seqs_tensor,
                   'skill_seqs_tensor': skill_seqs_tensor,
                   'time_lag_seqs_tensor': time_lag_seqs_tensor,
                   'correct_seqs_tensor': correct_seqs_tensor,
                   'timestamp_seqs_tensor': timestamp_seqs_tensor,
                   'seqs_length': seqs_length}
    return return_dict


class CTLSTMDataset_for_LPKT(Dataset):

    def __init__(self,
                 dataset='Assistment',
                 mode='train',
                 max_length=100,
                 cross_validation=False,
                 k_fold_num=0):

        assert mode == 'train' or mode == 'test' or mode == 'val'

        if cross_validation == False:
            self.file_path = 'data' + '/' + dataset + '/' + 'data_information_' + str(max_length) + '.pkl'

        else:
            self.file_path = 'data/' + dataset + '/data_information_' + str(max_length) + '_' + str(k_fold_num) + '.pkl'

        pkl_file = open(self.file_path, 'rb')

        data=pd.read_pickle(pkl_file)
        pkl_file.close()

        data_use = data.data_df[mode]

        self.len = len(data_use)

        self.time_lag_seq = data_use['time_lag']
        self.timestamp = data_use['timestamp']
        self.problem_seq = data_use['problem_seq']
        self.skill_seq = data_use['skill_seq']
        self.user_id = data_use['user_id']
        self.correct_seq = data_use['correct_seq']
        self.timeinterval_seq = data_use['timeinterval_seq']
        self.answertime_seq = data_use['answertime_seq']


    def __len__(self):
        return self.len

    def __getitem__(self, item):

        batch_dict = {
            'user_id': self.user_id[item],
            'time_lag_seq': self.time_lag_seq[item],
            'timestamp_seq': self.timestamp[item],
            'problem_seq': self.problem_seq[item],
            'skill_seq': self.skill_seq[item],
            'correct_seq': self.correct_seq[item],
            'timeinterval_seq': self.timeinterval_seq[item],
            'answertime_seq': self.answertime_seq[item],

        }

        return batch_dict


def pad_batch_fn_for_LPKT(many_batch_dict):

    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['problem_seq']), reverse=True)
    problem_seqs = [torch.LongTensor(seq['problem_seq']) for seq in sorted_batch]
    skill_seqs = [torch.LongTensor(seq['skill_seq']) for seq in sorted_batch]
    time_lag_seqs = [torch.FloatTensor(seq['time_lag_seq']) for seq in sorted_batch]
    correct_seqs = [torch.LongTensor(seq['correct_seq']) for seq in sorted_batch]
    timestamp_seqs = [torch.FloatTensor(seq['timestamp_seq']) for seq in sorted_batch]
    timeinterval_seqs = [torch.LongTensor(seq['timeinterval_seq']) for seq in sorted_batch]
    answertime_seqs = [torch.LongTensor(seq['answertime_seq']) for seq in sorted_batch]
    user_ids = [seq['user_id'] for seq in sorted_batch]


    seqs_length = torch.LongTensor(list(map(len, skill_seqs)))

    user_ids_tensor = torch.zeros(len(sorted_batch),).long()
    problem_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    skill_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    time_lag_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
    correct_seqs_tensor = torch.full((len(sorted_batch), seqs_length.max()), -1).long()
    timestamp_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
    timeinterval_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    answertime_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()

    for idx, (
    user_id, problem_seq, skill_seq, correct_seq, time_lag_seq, timestamp_seq, timeinterval_seq, answertime_seq,
    seq_len) \
            in enumerate(
        zip(user_ids, problem_seqs, skill_seqs, correct_seqs, time_lag_seqs, timestamp_seqs, timeinterval_seqs,
            answertime_seqs, seqs_length)):

        user_ids_tensor[idx] = user_id
        problem_seqs_tensor[idx, :seq_len] = torch.LongTensor(problem_seq)
        skill_seqs_tensor[idx, :seq_len] = torch.LongTensor(skill_seq)
        time_lag_seqs_tensor[idx, :seq_len] = torch.FloatTensor(time_lag_seq)
        correct_seqs_tensor[idx, :seq_len] = torch.LongTensor(correct_seq)
        timestamp_seqs_tensor[idx, :seq_len] = torch.FloatTensor(timestamp_seq)
        timeinterval_seqs_tensor[idx, :seq_len] = torch.LongTensor(timeinterval_seq)
        answertime_seqs_tensor[idx, :seq_len] = torch.LongTensor(answertime_seq)

    return_dict = {'user_id': user_ids_tensor,
                   'problem_seqs_tensor': problem_seqs_tensor,
                   'skill_seqs_tensor': skill_seqs_tensor,
                   'time_lag_seqs_tensor': time_lag_seqs_tensor,
                   'correct_seqs_tensor': correct_seqs_tensor,
                   'timestamp_seqs_tensor': timestamp_seqs_tensor,
                   'timeinterval_seqs_tensor': timeinterval_seqs_tensor,
                   'answertime_seqs_tensor': answertime_seqs_tensor,
                   'seqs_length': seqs_length}
    return return_dict


def pad_batch_fn_for_DKTForgetting(many_batch_dict):

    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['problem_seq']), reverse=True)
    problem_seqs = [torch.LongTensor(seq['problem_seq']) for seq in sorted_batch]
    skill_seqs = [torch.LongTensor(seq['skill_seq']) for seq in sorted_batch]
    correct_seqs = [torch.LongTensor(seq['correct_seq']) for seq in sorted_batch]
    timestamp_seqs = [torch.FloatTensor(seq['timestamp_seq']) for seq in sorted_batch]
    sequence_time_lag_seqs, repeated_time_lag_seqs, past_trial_counts_seqs = get_time_features(skill_seqs,
                                                                                               timestamp_seqs)
    repeated_time_lag_seqs = torch.from_numpy(repeated_time_lag_seqs)
    past_trial_counts_seqs = torch.from_numpy(past_trial_counts_seqs)
    sequence_time_lag_seqs = torch.from_numpy(sequence_time_lag_seqs)
    repeated_time_lag_seqs = torch.as_tensor(repeated_time_lag_seqs, dtype=torch.float32)
    past_trial_counts_seqs = torch.as_tensor(past_trial_counts_seqs, dtype=torch.float32)
    sequence_time_lag_seqs = torch.as_tensor(sequence_time_lag_seqs, dtype=torch.float32)
    seqs_length = torch.LongTensor(list(map(len, skill_seqs)))
    problem_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    skill_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    correct_seqs_tensor = torch.full((len(sorted_batch), seqs_length.max()), -1).long()
    timestamp_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
    for idx, (problem_seq, skill_seq,
              correct_seq, timestamp_seq,
              seq_len) in enumerate(
        zip(problem_seqs, skill_seqs,
            correct_seqs, timestamp_seqs,
            seqs_length)):
        problem_seqs_tensor[idx, :seq_len] = torch.LongTensor(problem_seq)
        skill_seqs_tensor[idx, :seq_len] = torch.LongTensor(skill_seq)
        correct_seqs_tensor[idx, :seq_len] = torch.LongTensor(correct_seq)
        timestamp_seqs_tensor[idx, :seq_len] = torch.FloatTensor(timestamp_seq)


    return_dict = {'problem_seqs_tensor': problem_seqs_tensor,
                   'skill_seqs_tensor': skill_seqs_tensor,

                   'correct_seqs_tensor': correct_seqs_tensor,
                   'timestamp_seqs_tensor': timestamp_seqs_tensor,
                   'repeated_time_lag_seqs_tensor': repeated_time_lag_seqs,
                   'sequence_time_lag_seqs_tensor': sequence_time_lag_seqs,
                   'past_trial_counts_seqs_tensor': past_trial_counts_seqs,
                   'seqs_length': seqs_length}
    return return_dict


def get_time_features(skill_seqs, timestamp_seqs):

    skill_max = max([max(i) for i in skill_seqs])
    inner_max_len = max(map(len, skill_seqs))
    repeated_time_lag_seq = np.zeros([len(skill_seqs), inner_max_len, 1], np.double)
    sequence_time_lag_seq = np.zeros([len(skill_seqs), inner_max_len, 1], np.double)
    past_trial_counts_seq = np.zeros([len(skill_seqs), inner_max_len, 1], np.double)
    for i in range(len(skill_seqs)):
        last_time = None
        skill_last_time = [None for _ in range(skill_max)]
        skill_cnt = [0 for _ in range(skill_max)]
        for j in range(len(skill_seqs[i])):
            sk = skill_seqs[i][j] - 1
            ti = timestamp_seqs[i][j]

            if skill_last_time[sk] is None:
                repeated_time_lag_seq[i][j][0] = 0
            else:
                repeated_time_lag_seq[i][j][0] = ti - skill_last_time[sk]
            skill_last_time[sk] = ti

            if last_time is None:
                sequence_time_lag_seq[i][j][0] = 0
            else:
                sequence_time_lag_seq[i][j][0] = (ti - last_time)
            last_time = ti

            past_trial_counts_seq[i][j][0] = (skill_cnt[sk])
            skill_cnt[sk] += 1

    repeated_time_lag_seq[repeated_time_lag_seq < 0] = 1
    sequence_time_lag_seq[sequence_time_lag_seq < 0] = 1
    repeated_time_lag_seq[repeated_time_lag_seq == 0] = 1e4
    sequence_time_lag_seq[sequence_time_lag_seq == 0] = 1e4
    past_trial_counts_seq += 1
    sequence_time_lag_seq *= 1.0 / 60
    repeated_time_lag_seq *= 1.0 / 60

    sequence_time_lag_seq = np.log(sequence_time_lag_seq)
    repeated_time_lag_seq = np.log(repeated_time_lag_seq)
    past_trial_counts_seq = np.log(past_trial_counts_seq)

    return sequence_time_lag_seq, repeated_time_lag_seq, past_trial_counts_seq

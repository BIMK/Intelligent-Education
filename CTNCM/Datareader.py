# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# class Datareader(object):
#
#     def __init__(self, train_per=0.7, val_per=0.1, test_per=0.2, max_length=100, path='data/Junyi/temp.csv', sep='\t'):
#         assert train_per + val_per + test_per == 1
#
#         self.train_per = train_per
#         self.val_per = val_per
#         self.test_per = test_per
#         self.max_length = int(max_length)
#         self.path = path
#         self.sep = sep
#
#         self.data_df = {
#             'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()
#         }
#
#         self.inter_df = pd.read_csv(self.path, sep=self.sep)
#         user_wise_dict = dict()
#         cnt, n_inters = 0, 0
#
#         for user, user_df in self.inter_df.groupby('user_id'):
#             user_df.sort_values(by='timestamp', inplace=True)
#             df = user_df[:self.max_length]  # consider the first 100 interactions
#             user_wise_dict[cnt] = {
#                 'user_id': cnt+1,
#                 'skill_seq': df['skill_id'].values.tolist(),
#                 'problem_seq': df['problem_id'].values.tolist(),
#                 'timestamp': df['timestamp'].values.tolist(),
#                 'correct_seq': df['correct'].values.tolist()
#             }
#             user_wise_dict[cnt]['time_lag'] = \
#                 [0.] + list(map(lambda x: float('%.2f' % (x[0] - x[1])), zip(user_wise_dict[cnt]['timestamp'][1:], user_wise_dict[cnt]['timestamp'][:-1])))
#             cnt += 1
#             n_inters += len(df)
#
#         self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')  #
#         self.n_skills = int(max(self.inter_df['skill_id']))
#         self.n_problems = int(max(self.inter_df['problem_id']))
#         self.n_users = int(len(self.inter_df[['user_id']].drop_duplicates()))
#         self.sum_length = [len(value.iloc[0]) for index, value in self.user_seq_df[['skill_seq']].iterrows()]
#
#     def show_columns(self):
#         return self.user_seq_df.columns
#
#     def devide_data(self):
#
#         n_examples = len(self.user_seq_df)
#         train_number = int(n_examples * self.train_per)
#         val_number = int(n_examples * self.val_per)
#         test_number = n_examples - train_number - val_number
#
#         self.data_df['train'] = self.user_seq_df.iloc[: train_number]
#         self.data_df['val'] = self.user_seq_df.iloc[train_number: train_number + val_number]
#         self.data_df['val'].index = range(len(self.data_df['val']))
#         self.data_df['test'] = self.user_seq_df.iloc[train_number + val_number:]
#         self.data_df['test'].index = range(len(self.data_df['test']))



def Devide_Fold_and_Save(dataset, max_length, cross_validation, k_fold):

    path = 'data' + '/' + dataset + '/' + 'interactions.csv'

    if cross_validation < 2:
        print("prepare data for no cross validation...")
        print('print("WRONG! Please set cross_validation = True")')
        return 0

    elif cross_validation >= 2:
        assert k_fold >= 1
        print("prepare data for {}-fold cross validation...".format(k_fold))
        data = Datareader_fold(path=path, max_length=max_length)
        for count in tqdm(range(k_fold)):
            fold_begin, fold_end = data.get_fold_position()
            data.devide_data(fold_begin_num=fold_begin[count], fold_end_num=fold_end[count])
            save_train_path = {}
            save_train_path = 'data/' + dataset + '/TrainSet/train' + str(count) + '_' + str(data.max_length) + '.csv'
            save_val_path = 'data/' + dataset + '/ValSet/val' + str(count) + '_' + str(data.max_length) + '.csv'
            save_test_path = 'data/' + dataset + '/TestSet/test' + str(count) + '_' + str(data.max_length) + '.csv'

            data.data_df['train'].to_csv(save_train_path)
            data.data_df['val'].to_csv(save_val_path)
            data.data_df['test'].to_csv(save_test_path)
        setting_dict = {'n_users': data.n_users,
                        'n_skills': data.n_skills,
                        'n_problems': data.n_problems,
                        'n_examples': data.n_examples}
        json_str = json.dumps(setting_dict)
        json_path = 'data/' + dataset + '/settings.json'
        with open(json_path, 'w') as json_file:
            json_file.write(json_str)


class Datareader_fold(object):
    # This is Datareader function can adaptive for LPKT's special features
    def __init__(self,
                 max_length,
                 cross_validation=True,
                 k_fold=5,
                 path='data/Assistment12/interactions.csv',
                 sep='\t'):
        assert cross_validation == True  # 此类为 k 折交叉验证
        assert k_fold >= 1
        self.max_length = int(max_length)
        self.path = path
        self.sep = sep
        self.k_fold = k_fold
        self.unit_time_lag = 1

        self.data_df = {
            'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()
        }

        self.inter_df = pd.read_csv(self.path, sep=self.sep)

        user_wise_dict = dict()
        cnt, n_inters = 0, 0
        for user, user_df in self.inter_df.groupby('user_id'):
            user_df.sort_values(by='timestamp', inplace=True)
            df = user_df[ :self.max_length]
            user_wise_dict[cnt] = {
                'user_id': cnt + 1,
                'skill_seq': df['skill_id'].values.tolist(),
                'problem_seq': df['problem_id'].values.tolist(),
                'timestamp': df['timestamp'].values.tolist(),
                'correct_seq': df['correct'].values.tolist(),
            }

            user_wise_dict[cnt]['time_lag'] =\
                [0.] + list(map(lambda x: float('%.2f' % ((x[0] - x[1]) * self.unit_time_lag) ), zip(user_wise_dict[cnt]['timestamp'][1:], user_wise_dict[cnt]['timestamp'][:-1])))

            cnt += 1
            n_inters += len(df)

        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')
        self.n_users = len(self.inter_df[['user_id']].drop_duplicates())
        self.n_skills = int(max(self.inter_df['skill_id'])) + 1
        self.n_problems = int(max(self.inter_df['problem_id'])) + 1


        self.n_examples = len(self.user_seq_df)
        self.sum_length = [len(value.iloc[0]) for index, value in self.user_seq_df[['skill_seq']].iterrows()]

    def show_columns(self):
        return self.user_seq_df.columns

    def get_fold_position(self):
        fold_size = int(self.n_examples/self.k_fold)
        fold_begin = [i * fold_size for i in range(self.k_fold)]
        fold_end = [(i + 1) * fold_size for i in range(self.k_fold)]
        fold_end[-1] = self.n_examples

        return fold_begin, fold_end

    def devide_data(self, fold_begin_num, fold_end_num):

        self.data_df['test'] = self.user_seq_df.iloc[fold_begin_num: fold_end_num]
        self.data_df['test'].index = range(len(self.data_df['test']))
        residual_df = pd.concat([self.user_seq_df.iloc[0: fold_begin_num], self.user_seq_df.iloc[fold_end_num:self.n_examples]])
        dev_size = int(0.1 * len(residual_df))
        np.random.seed(2021)
        dev_indices = np.random.choice(residual_df.index, dev_size, replace=False)
        self.data_df['val'] = self.user_seq_df.iloc[dev_indices]
        self.data_df['val'].index = range(len(self.data_df['val']))
        self.data_df['train'] = residual_df.drop(dev_indices)
        self.data_df['train'].index = range(len(self.data_df['train']))
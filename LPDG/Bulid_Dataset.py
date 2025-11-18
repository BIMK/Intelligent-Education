import os
import torch
import argparse
from random import shuffle
from sequential.seq2pat import Seq2Pat
import pandas as pd
from utils import get_dataset_information
from tqdm import tqdm

def set_seed(seed):
    import  numpy as np
    import  random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def increment_sequences(sequences):
    incremented_sequences = [[x + 1 for x in seq] for seq in sequences]
    return incremented_sequences

def is_sublist(sublst, lst):
    for element in sublst:
        try:
            ind = lst.index(element)
        except ValueError:
            return False
        lst = lst[ind + 1:]
    return True

def seq2regeard(sequences, problem_num, problem2skill):
    problem_seq_list = []
    skill_seq_list = []
    correct_seq_list = []
    for ele in sequences:
        if ele > problem_num:
            problem_seq_list.append(ele - problem_num-1)
            skill_seq_list.append(problem2skill[str(int(ele - problem_num-1))])
            correct_seq_list.append(1)
        else:
            problem_seq_list.append(ele-1)
            skill_seq_list.append(problem2skill[str(int(ele-1))])
            correct_seq_list.append(0)
    return problem_seq_list, skill_seq_list, correct_seq_list

if __name__ == '__main__':
    set_seed(2026)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./dataset/Assistment09',
                        help='The path to the training dataset.')
    parser.add_argument('--n_jobs', type=int, default=3, help='The job number for Seq2Pat pattern mining.')
    import json

    seq2set=set()
    args = parser.parse_args()
    datasetconfig = json.load("./config/DataConfig.json")
    datasetname =args.root_path.split("/")[-1]
    args.dataset=datasetname
    args.min_frequency=datasetconfig[datasetname]["min_frequency"]
    args.max_span=datasetconfig[datasetname]["max_span"]
    max_seq_len = 50
    min_seq_len = 10
    all_interaction_data = os.path.join(args.root_path, 'all_interactions.csv')
    all_interaction_data_df = pd.read_csv(all_interaction_data)

    setting_path = os.path.join(args.root_path, 'settings.json')
    dataset = args.root_path.split('/')[-1]
    dataset_information = get_dataset_information(dataset=dataset, max_length=['max_length'], path=setting_path)
    problem_num = int(dataset_information['problem_num'])

    problem_seq_list = all_interaction_data_df['problem_seq'].tolist()
    correct_seq_list = all_interaction_data_df['correct_seq'].tolist()
    problem_seq_list = [list(map(int, seq.strip('[').strip(']').split(','))) for seq in problem_seq_list]
    correct_seq_list = [list(map(int, seq.strip('[').strip(']').split(','))) for seq in correct_seq_list]
    problem_seq_list = increment_sequences(problem_seq_list)

    all_interaction = []
    for index, ele in enumerate(problem_seq_list):
        s = zip(problem_seq_list[index], correct_seq_list[index])
        seq = []
        for j in s:
            seq.append(j[0] + j[1] * problem_num)
        all_interaction.append(seq)

    problem2skillfile=os.path.join(args.root_path,'problem2skill.json')

    with open(problem2skillfile, 'r') as f:
        problem2skill = json.load(f)
    output_path = os.path.join(args.root_path, 'pair_p_s_c.pth')
    if not os.path.exists(output_path):
        print('Performing rule-based pattern-mining!')
        seq2pat = Seq2Pat(
            sequences=all_interaction,
            n_jobs=args.n_jobs,
            max_span=args.max_span,
        )
        print("Pattern mining!")
        patterns = seq2pat.get_patterns(min_frequency=args.min_frequency)
        patterns_value = []
        for pattern in tqdm(patterns):
            pattern=pattern[:-1]
            if len(pattern)<10:
                continue
            if str(pattern) in seq2set:
                continue
            else:
                patterns_value.append(pattern)
                seq2set.add(str(pattern))
        print(len(patterns_value))
        print(f'Rule-based patterns mined with size {len(patterns_value)}')
        train_data_path = os.path.join(args.root_path, 'TrainSet/train_data.csv')
        train_data = pd.read_csv(train_data_path)
        train_problem_seq_list = train_data['problem_seq'].tolist()
        train_correct_seq_list = train_data['correct_seq'].tolist()
        train_problem_seq_list = [list(map(int, seq.strip('[').strip(']').split(','))) for seq in train_problem_seq_list]
        train_correct_seq_list = [list(map(int, seq.strip('[').strip(']').split(','))) for seq in train_correct_seq_list]

        train_problem_seq_list = increment_sequences(train_problem_seq_list)
        print('Pre-processing the extracted patterns for dataset regeneration.')
        train_seq = []
        for index, ele in enumerate(train_problem_seq_list):
            s = zip(train_problem_seq_list[index], train_correct_seq_list[index])
            seq = []
            for j in s:
                seq.append(j[0] + j[1] * problem_num)
            train_seq.append(seq)
        data_generation_pair = []
        i=0
        for seq_ori in tqdm(train_seq):
            shuffle(patterns_value)
            cnt = 0
            for p in patterns_value:
                if cnt == 5 :
                    break
                if is_sublist(p, seq_ori):
                    d = {}
                    t = seq2regeard(seq_ori, problem_num, problem2skill)
                    d["ori_problem"] = t[0]
                    d['ori_skill'] = t[1]
                    d["ori_correct"] = t[2]
                    t = seq2regeard(p, problem_num, problem2skill)
                    d['model_problem'] = t[0]
                    d['model_skill'] = t[1]
                    d['model_correct'] = t[2]
                    data_generation_pair.append(d)
                    cnt += 1
                else:
                    i+=1
        print(i)
        print(f'Building sequence-pattern pair dataset with size {len(data_generation_pair)}.')

        torch.save(data_generation_pair, output_path)
        print("预训练数据集生成完成")

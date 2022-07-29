# -*-coding:utf-8 -*-

import os
import json
import time


def get_Average_optimal_effect(k_fold_optimal_effect):

    train_acc = [ i['train_acc'] for i in k_fold_optimal_effect]
    train_auc = [ i['train_auc'] for i in k_fold_optimal_effect]
    val_acc = [ i['val_acc'] for i in k_fold_optimal_effect]
    val_auc = [ i['val_auc'] for i in k_fold_optimal_effect]
    test_acc = [ i['test_acc'] for i in k_fold_optimal_effect]
    test_auc = [ i['test_auc'] for i in k_fold_optimal_effect]

    Average_optimal_effect={'train_acc': sum(train_acc)/len(train_acc),
                            'train_auc': sum(train_auc)/len(train_auc),
                            'val_acc': sum(val_acc)/len(val_acc),
                            'val_auc': sum(val_auc)/len(val_auc),
                            'test_acc': sum(test_acc)/len(test_acc),
                            'test_auc': sum(test_auc)/len(test_auc),}

    return Average_optimal_effect

def get_dataset_information(dataset, max_length, path):
    print('path: ', path)
    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)
    skill_num = load_dict['n_skills']
    problem_num = load_dict['n_problems']
    sequence_num = load_dict['n_users']

    print('sequence_num: ', sequence_num)
    print('skill_num: ', skill_num)
    print('problem_num: ', problem_num)


    return_dict = {'dataset': dataset,
                   'skill_num': skill_num,
                   'problem_num': problem_num,
                   'sequence_num': sequence_num
                   }
    return return_dict

def initialize_effect():
    effect = {'train_loss': list(),
              'train_acc': list(),
              'train_auc': list(),
              'val_loss': list(),
              'val_acc': list(),
              'val_auc': list(),
              'test_loss': list(),
              'test_acc': list(),
              'test_auc': list()}
    return effect

def get_optimal_value(effect):
    epoch = effect['val_auc'].index(max(effect['val_auc']))
    optimal_effect = {'epoch': epoch+1,
                      'train_acc': effect['test_acc'][epoch],
                      'train_auc': effect['train_auc'][epoch],
                      'val_acc': effect['val_acc'][epoch],
                      'val_auc': effect['val_auc'][epoch],
                      'test_acc': effect['test_acc'][epoch],
                      'test_auc': effect['test_auc'][epoch]}

    return optimal_effect

def get_Data_Time():
    localtime = time.asctime(time.localtime(time.time()))
    data_time = localtime.split(' ')
    data_time = [i for i in data_time if i != '']
    data_time[3] = '_'.join(data_time[3].split(':'))
    data_time = '_'.join(data_time[:-1])
    return data_time


import os

import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from myDataloader import CTLSTMDataset_for_LPKT, pad_batch_fn_for_LPKT
from Datareader import Datareader_fold, Devide_Fold_and_Save
import time
from Model_list import LPKT
import pickle
from tqdm import tqdm
import time
import json
# from build_graph import *

def fold_ready_train_student(settings):
    if settings['cross_validation'] == False:
        path_test = 'data/' + settings['dataset'] + '/data_information_' + str(settings['max_length']) + '.pkl'
        if os.path.exists(path_test) == False:
            Devide_Fold_and_Save(settings)

        dataset_information = get_dataset_information(dataset=dataset, max_length=max_length, path=path_test)
        settings['skill_num'] = int(dataset_information['skill_num'])
        settings['problem_num'] = int(dataset_information['problem_num'])
        settings['student_num'] = int(dataset_information['student_num'])
        settings['n_it'] = int(dataset_information['n_it'])
        settings['n_at'] = int(dataset_information['n_at'])
        settings['dataset'] = dataset_information['dataset']
        settings['student_num'] = int(dataset_information['student_num'])

        optimal_effect = train_student(settings, k_fold_num=0)
        print("optimal_effect: ")
        for item in optimal_effect.items():
            print(item)

    elif settings['cross_validation'] == True:
        assert settings['k_fold'] >= 1
        k_fold_optimal_effect = list()

        path_test = 'data/' + settings['dataset'] + '/data_information_' + str(settings['max_length']) + '_' + str(
            1) + '.pkl'
        if os.path.exists(path_test) == False:
            Devide_Fold_and_Save(settings, model_name=settings['model'])

        for k_fold_num in range(settings['k_fold']):
            path = 'data/' + settings['dataset'] + '/data_information_' + str(settings['max_length']) + '_' + str(
                k_fold_num) + '.pkl'
            if os.path.exists(path) == True:
                dataset_information = get_dataset_information(dataset=dataset, max_length=max_length, path=path)
                settings['skill_num'] = int(dataset_information['skill_num'])
                settings['problem_num'] = int(dataset_information['problem_num'])
                settings['student_num'] = int(dataset_information['student_num'])
                settings['n_it'] = int(dataset_information['n_it'])
                settings['n_at'] = int(dataset_information['n_at'])
                settings['dataset'] = dataset_information['dataset']
                print('******* No.{}-fold cross validation *******'.format(k_fold_num + 1))
                optimal_effect = train_student(settings, k_fold_num)
                k_fold_optimal_effect.append(optimal_effect)
            elif os.path.exists(path) == False:
                print("Dataset configuration failed, please try again.")

        Average_optimal_effect = get_Average_optimal_effect(k_fold_optimal_effect)
        print("******* Average optimal effect after {}-fold cross validation *******".format(settings['k_fold']))
        for item in Average_optimal_effect.items():
            print(item)


def get_Average_optimal_effect(k_fold_optimal_effect):

    val_acc = [i['val_acc'] for i in k_fold_optimal_effect]
    val_auc = [i['val_auc'] for i in k_fold_optimal_effect]
    test_acc = [i['test_acc'] for i in k_fold_optimal_effect]
    test_auc = [i['test_auc'] for i in k_fold_optimal_effect]

    Average_optimal_effect = {'val_acc': sum(val_acc) / len(val_acc),
                              'val_auc': sum(val_auc) / len(val_auc),
                              'test_acc': sum(test_acc) / len(test_acc),
                              'test_auc': sum(test_auc) / len(test_auc)}

    return Average_optimal_effect


def get_dataset_information(dataset='Junyi', max_length=100, path='data/Junyi/data_information_100.pkl'):
    path = path
    pkl_file = open(path, 'rb')
    data = pd.read_pickle(pkl_file)
    pkl_file.close()

    skill_num = data.n_skills
    problem_num = data.n_problems
    student_num = data.n_users
    n_it = data.n_it
    n_at = data.n_at

    print('student_num: ', student_num)
    print('skill_num: ', skill_num)
    print('problem_num: ', problem_num)
    print('n_it: ', n_it)
    print('n_at: ', n_at)

    return_dict = {'dataset': dataset,
                   'skill_num': skill_num,
                   'problem_num': problem_num,
                   'student_num': student_num,
                   'n_at': n_at,
                   'n_it': n_it}
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


def train_student(settings, k_fold_num):
    """Training process."""
    batch_size = settings['batch_size']
    epoch_num = settings['epoch_num']
    effect = initialize_effect()
    effect['setting'] = settings.copy()
    train_data = CTLSTMDataset_for_LPKT(dataset=settings['dataset'],
                                        mode='train',
                                        max_length=settings['max_length'],
                                        cross_validation=settings['cross_validation'],
                                        k_fold_num=k_fold_num)
    val_data = CTLSTMDataset_for_LPKT(dataset=settings['dataset'],
                                      mode='val',
                                      max_length=settings['max_length'],
                                      cross_validation=settings['cross_validation'],
                                      k_fold_num=k_fold_num)
    test_data = CTLSTMDataset_for_LPKT(dataset=settings['dataset'],
                                       mode='test',
                                       max_length=settings['max_length'],
                                       cross_validation=settings['cross_validation'],
                                       k_fold_num=k_fold_num)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                  drop_last=True, num_workers=0, collate_fn=pad_batch_fn_for_LPKT)
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=False,
                                drop_last=True, num_workers=2, collate_fn=pad_batch_fn_for_LPKT)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False,
                                 drop_last=True, num_workers=2, collate_fn=pad_batch_fn_for_LPKT)
    model_name = eval('{0}.{0}'.format(settings['model']))
    model = model_name(settings)
    model = model.to(device)
    optimzer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    Model_Save_Path = list()
    for epoch in range(1, epoch_num + 1):
        epoch_train_loss = 0.0
        print('Epoch.{} starts.'.format(epoch))
        current_time = time.time()
        predictions_list = list()
        labels_list = list()
        model.train()
        for index, log_dict in enumerate(tqdm(train_dataloader)):
            optimzer.zero_grad()
            output_dict = model.forward(log_dict)
            predictions_list.append(torch.squeeze(output_dict['predictions'].detach()))
            labels_list.append(torch.squeeze(output_dict['labels'].detach()))
            loss = model.loss(output_dict)
            loss.backward()
            optimzer.step()
            epoch_train_loss += loss.item()
        print("for train ", "loss:", epoch_train_loss, "metrics: none")
        effect['train_loss'].append(epoch_train_loss)
        save_path = save_snapshot(model, settings, k_fold_num, epoch)
        Model_Save_Path.append(save_path)
        if len(Model_Save_Path) > 10:
            remove_path = Model_Save_Path.pop(0)
            os.remove(remove_path)

        ################################### 一轮训练完毕，验证模型 ##################################################
        epoch_val_loss = 0.0
        predictions_list = list()
        labels_list = list()
        model.eval()
        for index, log_dict in enumerate(val_dataloader):
            with torch.no_grad():
                output_dict = model.forward(log_dict)
                predictions_list.append(torch.squeeze(output_dict['predictions'].detach()))
                labels_list.append(torch.squeeze(output_dict['labels'].detach()))
                loss = model.loss(output_dict)
                epoch_val_loss += loss.item()

        metrics = get_metrics(predictions_list, labels_list)
        print("for val ", "loss:", epoch_val_loss, "metrics:", metrics)
        effect['val_loss'].append(epoch_val_loss)
        effect['val_acc'].append(metrics['acc'])
        effect['val_auc'].append(metrics['auc'])

        ################################### 测试模型 ##################################################
        predictions_list = list()
        labels_list = list()
        epoch_test_loss = 0.0
        model.eval()
        for index, log_dict in enumerate(test_dataloader):
            with torch.no_grad():
                output_dict = model.forward(log_dict)
                predictions_list.append(torch.squeeze(output_dict['predictions'].detach()))
                labels_list.append(torch.squeeze(output_dict['labels'].detach()))
                loss = model.loss(output_dict)
                epoch_test_loss += loss.item()
        metrics = get_metrics(predictions_list, labels_list)

        print("for test ", "loss:", epoch_test_loss, "metrics:", metrics)
        effect['test_loss'].append(epoch_test_loss)
        effect['test_acc'].append(metrics['acc'])
        effect['test_auc'].append(metrics['auc'])

        save_result(result_copy=effect.copy(), epoch=epoch, settings=settings, k_fold_num=k_fold_num)

        print("epoch_time: ", time.time() - current_time)

        if settings['early_stop'] == True and epoch > settings['early_stop_num']:
            val_auc_temp = effect['val_auc'][-settings['early_stop_num']:]
            max_val_auc_temp = max(val_auc_temp)
            if max_val_auc_temp == val_auc_temp[0]:
                print("epoch=", epoch, "early stop!")
                break
    optimal_effect = get_optimal_value(effect)
    print("optimal effect:")
    for item in optimal_effect.items():
        print(item)

    return optimal_effect


def get_optimal_value(effect):
    epoch = effect['val_auc'].index(max(effect['val_auc']))
    optimal_effect = {'epoch': epoch + 1,
                      'val_acc': effect['val_acc'][epoch],
                      'val_auc': effect['val_auc'][epoch],
                      'test_acc': effect['test_acc'][epoch],
                      'test_auc': effect['test_auc'][epoch]}

    return optimal_effect


def deal_predictions(output_dict, seqs_length, batch_size):
    predictions = torch.squeeze(output_dict['predictions'])
    predictions_tensor = torch.zeros(batch_size, seqs_length.max()).long()
    start = 0
    end = 0
    for index, length in enumerate(seqs_length):
        end += length
        predictions_tensor[index][:length] = predictions[start:end]
        start += length
    predictions_packed = torch.nn.utils.rnn.pack_padded_sequence(predictions_tensor[:, 1:], seqs_length - 1,
                                                                 batch_first=True)
    output_dict['predictions'] = predictions_packed.data

    return output_dict


def val_student(net, settings, val_dataloader):
    output_dict_list = list()
    epoch_val_loss = 0.0
    model_name = eval('{0}.{0}'.format(settings['model']))
    model = model_name(settings)
    model.load_state_dict(net.state_dict())

    model = model.to(device)
    model.eval()
    for index, log_dict in enumerate(val_dataloader):
        with torch.no_grad():
            output_dict = model.forward(log_dict)
            output_dict_list.append(output_dict)
            loss = model.loss(output_dict)
            epoch_val_loss += float('%.8f' % loss.detach().cpu())

    return output_dict_list, epoch_val_loss


def test_student(net, settings, test_dataloader):
    output_dict_list = list()
    epoch_test_loss = 0.0
    model_name = eval('{0}.{0}'.format(settings['model']))
    model = model_name(settings)
    model.load_state_dict(net.state_dict())

    model = model.to(settings['device'])
    model.eval()

    for index, log_dict in enumerate(test_dataloader):
        with torch.no_grad():
            output_dict = model.forward(log_dict)
            output_dict_list.append(output_dict)
            loss = model.loss(output_dict)
            epoch_test_loss += float('%.8f' % loss.detach().cpu())

    return output_dict_list, epoch_test_loss


def get_metrics(prediction_list, label_list):
    predictions = torch.squeeze(torch.cat(prediction_list).cpu())
    predictions_round = torch.round(predictions)
    labels = torch.squeeze(torch.cat(label_list).cpu())

    accuracy = accuracy_score(labels, predictions_round)
    auc = roc_auc_score(labels, predictions)

    return_dict = {'acc': float('%.6f' % accuracy), 'auc': float('%.6f' % auc)}
    return return_dict


def get_Data_Time():
    localtime = time.asctime(time.localtime(time.time()))
    data_time = localtime.split(' ')
    data_time = [i for i in data_time if i != '']
    data_time[3] = '_'.join(data_time[3].split(':'))
    data_time = '_'.join(data_time[:-1])
    return data_time


def save_snapshot(model, settings, k_fold_num, epoch):
    Model_Save_Path = 'Model_Save/' + settings['model'] + '/' + settings['dataset'] + '_cross_' + \
                      str(settings['cross_validation']) + '_fold_' + str(k_fold_num) + \
                      '_epoch' + str(epoch) + '_' + settings['data_time']
    f = open(Model_Save_Path, 'wb')
    torch.save(model.state_dict(), f)
    f.close()
    return Model_Save_Path


def save_effect(effect, settings, k_fold_num):
    save_path = 'Model_Save/' + settings['model'] + '/' + settings['dataset'] + '_cross_' + str(
        settings['cross_validation']) + \
                '_Effect_fold' + str(k_fold_num) + \
                settings['data_time']
    np.save(save_path, effect)


def save_result(result_copy, epoch, settings, k_fold_num):
    save_path = 'Result_save/' + settings['model'] + '/' + settings['dataset'] + \
                '_fold' + str(k_fold_num) + \
                settings['data_time'] + '.json'
    if epoch == 1:
        result_copy['setting'].pop('device')
    effect_string = json.dumps(result_copy, indent=4)
    with open(save_path, 'w') as json_file:
        json_file.write(effect_string)


if __name__ == "__main__":

    dataset = 'Assistment12'
    # dataset = 'Assistment17'
    # dataset = 'Junyi'
    # dataset = 'Slepemapy'
    # dataset = 'Assistment17_50'
    # dataset = 'Junyi_Whole'


    model = 'HD-LPKT'
    cuda = 'cuda:1'
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    max_length = 50
    batch_size = 64
    hidden_size = 64
    embed_size = 64
    epoch_num = 100
    early_stop = True
    early_stop_num = 5
    cross_validation = True
    k_fold = 5
    data_time = get_Data_Time()
    model_list = [model]
    assert model_list[0] == 'LPKT'

    d_k = 128
    d_e = 128
    d_a = 50
    dropout = 0.2

    for model_name in model_list:
        model = model_name
        settings = {
            'device': device,
            'dataset': dataset,
            'model': model,
            'hidden_size': hidden_size,
            'embed_size': embed_size,
            'max_length': max_length,
            'batch_size': batch_size,
            'epoch_num': epoch_num,
            'early_stop': early_stop,
            'early_stop_num': early_stop_num,
            'cross_validation': cross_validation,
            'k_fold': k_fold,
            'data_time': data_time,
            'd_k': d_k,
            'd_e': d_e,
            'd_a': d_a,
            'dropout': dropout
        }
        print("settings")
        for setting in settings.items():
            print(setting)

        fold_ready_train_student(settings)

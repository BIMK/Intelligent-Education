# -*-coding:utf-8 -*-


import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from myDataloader import CTLSTMDataset, pad_batch_fn
from Datareader import Devide_Fold_and_Save
from utils import get_Average_optimal_effect, initialize_effect, get_dataset_information, \
    get_optimal_value, get_Data_Time

from Model_list import CT_NCM
from tqdm import tqdm
import time
# import json
import argparse
import random





def fold_ready_train_student(params, dataset):
    if params.cross_validation == False:
        print("Please set cross_validation = True")

    elif params.cross_validation >=1:
        assert params.k_fold >= 1
        k_fold_optimal_effect = list()

        path_test = 'data/' + dataset + '/TrainSet/train' + str(0) + '_' + str(params.max_length) + '.csv'
        if os.path.exists(path_test) == False:
            # Make data
            Devide_Fold_and_Save(dataset=dataset, max_length=params.max_length, cross_validation=params.cross_validation, k_fold=params.k_fold)


        for k_fold_num in range(params.k_fold):
            path = 'data/' + dataset + '/settings.json'

            if os.path.exists(path) == True:
                dataset_information = get_dataset_information(dataset=dataset, max_length=['max_length'], path=path)
                params.skill_num = int(dataset_information['skill_num'])
                params.problem_num = int(dataset_information['problem_num'])
                params.sequence_num = int(dataset_information['sequence_num'])
                params.datatime = get_Data_Time()

                print('******* No.{}-fold cross validation *******'.format(k_fold_num + 1))
                optimal_effect = train_student(k_fold_num)
                k_fold_optimal_effect.append(optimal_effect)
            elif os.path.exists(path) == False:
                print("Dataset configuration failed, please try again.")

        Average_optimal_effect = get_Average_optimal_effect(k_fold_optimal_effect)
        print("******* Average optimal effect after {}-fold cross validation *******".format(params.k_folds))
        for item in Average_optimal_effect.items():
            print(item)



def train_student(k_fold_num):
    assert params.model == 'CT_NCM'

    batch_size = params.batch_size
    epoch_num = params.epoch_num
    effect = initialize_effect()

    train_data = CTLSTMDataset(dataset=params.dataset,
                               mode='train',
                               max_length=params.max_length,
                               cross_validation=params.cross_validation,
                               k_fold_num=k_fold_num)
    val_data = CTLSTMDataset(dataset=params.dataset,
                             mode='val',
                             max_length=params.max_length,
                             cross_validation=params.cross_validation,
                             k_fold_num=k_fold_num)
    test_data = CTLSTMDataset(dataset=params.dataset,
                              mode='test',
                              max_length=params.max_length,
                              cross_validation=params.cross_validation,
                              k_fold_num=k_fold_num)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, num_workers=0, collate_fn=pad_batch_fn)
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=False,
                                drop_last=False, num_workers=0, collate_fn=pad_batch_fn)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False,
                                 drop_last=False, num_workers=0, collate_fn=pad_batch_fn)

    print('params.model: ', params.model)
    model_name = eval('{0}.{0}'.format(params.model))


    model = model_name(dataset=params.dataset, skill_num=params.skill_num, problem_num=params.problem_num,
                       device=params.device, hidden_size=params.hidden_size, embed_size=params.embed_size,
                       prelen1=params.prelen1, prelen2=params.prelen2, dropout1=params.dropout1, dropout2=params.dropout2)


    model = model.to(params.device)
    optimzer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
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

        metrics = get_metrics(predictions_list, labels_list)

        print("for train ", "loss:", epoch_train_loss, "metrics: ", metrics)
        effect['train_loss'].append(epoch_train_loss)
        effect['train_acc'].append(metrics['acc'])
        effect['train_auc'].append(metrics['auc'])

        # Save Model Parameters
        save_path = save_snapshot(model, model_name=params.model, dataset=params.dataset, k_fold_num=k_fold_num, epoch=epoch, datatime=params.datatime)

        train_str = 'For train. loss: {}, acc: {}, auc: {}'.format(str(epoch_train_loss), str(metrics['acc']), str(metrics['auc']))
        Model_Save_Path.append(save_path)

        # We only want to save the results of the last 10 iterations
        if len(Model_Save_Path) > 10:
            remove_path = Model_Save_Path.pop(0)
            os.remove(remove_path)

        ################################### After a round of training, verify the model ##################################################
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
        val_str = 'For val. loss: {}, acc: {}, auc: {}'.format(str(epoch_train_loss), str(metrics['acc']), str(metrics['auc']))

        ################################### After a round of training, test the model ##################################################
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
        test_str = 'For test. loss: {}, acc: {}, auc: {}'.format(str(epoch_train_loss), str(metrics['acc']), str(metrics['auc']))

        save_result(train_str, val_str, test_str, model=params.model, epoch=epoch, dataset=params.dataset, datatime=params.datatime, k_fold_num=k_fold_num)

        print("epoch_time: ", time.time() - current_time)

        if params.early_stop >= 2 and epoch > params.early_stop:
            val_auc_temp = effect['val_auc'][-params.early_stop:]
            max_val_auc_temp = max(val_auc_temp)
            if max_val_auc_temp == val_auc_temp[0]:
                print("epoch=", epoch, "early stop!")
                break

    optimal_effect = get_optimal_value(effect)
    print("optimal effect:")
    for item in optimal_effect.items():
        print(item)

    return optimal_effect

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_metrics(prediction_list, label_list):
    predictions = torch.squeeze(torch.cat(prediction_list).cpu())
    predictions_round = torch.round(predictions)
    labels = torch.squeeze(torch.cat(label_list).cpu())

    accuracy = accuracy_score(labels, predictions_round)
    auc = roc_auc_score(labels, predictions)

    return_dict = {'acc': float('%.6f' % accuracy), 'auc': float('%.6f' % auc)}
    return return_dict

def save_snapshot(model, model_name, dataset, k_fold_num, epoch, datatime):
    Model_Save_Path = 'Model_Save/' + model_name + '/' + dataset + '_fold_' + str(k_fold_num) +\
                      '_epoch' + str(epoch) + '_' + str(datatime)
    f = open(Model_Save_Path, 'wb')
    torch.save(model.state_dict(), f)
    f.close()
    return Model_Save_Path

def save_effect(effect, settings, k_fold_num):
    save_path = 'Model_Save/' + settings['model'] + '/' + settings['dataset'] + '_cross_' + str(settings['cross_validation']) + \
                '_Effect_fold' + str(k_fold_num) + \
                 settings['data_time']
    np.save(save_path, effect)

def save_result(train_str, val_str, test_str, epoch, model, dataset, datatime, k_fold_num):
    save_path = 'Result_save/'+ model + '/' + dataset + \
                '_fold' + str(k_fold_num) + \
                 datatime + '.txt'
    f = open(save_path, "a")
    f.write('Epoch.' + str(epoch) + '\n')
    f.write(train_str + '\n')
    f.write(val_str + '\n')
    f.write(test_str + '\n')
    f.close()




if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser()

    # Basic Parameters
    parser.add_argument('--dataset', type=str, default='Assistment17',
                        help='choose a dataset.')
    parser.add_argument('--model', type=str, default='CT_NCM', help='choose a model.')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose a device.')
    parser.add_argument('--cross_validation', type=int, default=5, help='cross validation')
    parser.add_argument('--max_length', type=int, default=100, help='choose a value for max length.')
    parser.add_argument('--epoch_num', type=int, default=100, help='maximum epoch num.')
    parser.add_argument('--early_stop', type=int, default=5, help='number of early stop for AUC.')
    parser.add_argument('--k_fold', type=int, default=5, help='number of folds for cross_validation.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')

    # Parameters For CT-NCM
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size.')
    parser.add_argument('--embed_size', type=int, default=64, help='embed_size.')
    parser.add_argument('--prelen1', type=int, default=256, help='the first-second layer of performance prediction.')
    parser.add_argument('--prelen2', type=int, default=128, help='the second-third layer of performance prediction.')
    parser.add_argument('--dropout1', type=float, default=0,
                        help='the first-second layer\'s dropout of performance prediction.')
    parser.add_argument('--dropout2', type=float, default=0,
                        help='the second-third layer\'s dropout of performance prediction.')


    params = parser.parse_args()
    dataset = params.dataset
    fold_ready_train_student(params, dataset)

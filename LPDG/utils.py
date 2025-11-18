import yaml
import random
import torch
import numpy as np
import os
import importlib
import torch.nn as nn
from copy import deepcopy
import json
torch.autograd.set_detect_anomaly(True)
import time
def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(config):
    if isinstance(config['device'], int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device'])
        config['device'] = 'cuda'

def get_Data_Time():
    localtime = time.asctime(time.localtime(time.time()))
    data_time = localtime.split(' ')
    data_time = [i for i in data_time if i != '']
    data_time[3] = '_'.join(data_time[3].split(':'))
    data_time = '_'.join(data_time[:-1])
    return data_time

def setup_environment(config):
    seed_everything(config['seed'])
    set_device(config)

def get_model_class(config):
    path = '.'.join(['model', config['model'].lower()])
    module = importlib.import_module(path, __name__)
    model_class = getattr(module, config['model'])
    return model_class

def prepare_datasets(config):
    model_class = get_model_class(config['model'])
    dataset_class = model_class._get_dataset_class(config)

    train_dataset = dataset_class(config, phase='train')
    val_dataset = dataset_class(config, phase='val')
    test_dataset = dataset_class(config, phase='test')

    train_dataset.build()
    val_dataset.build()
    test_dataset.build()

    return train_dataset, val_dataset, test_dataset

def prepare_model(config, dataset_list):
    model_class = get_model_class(config['model'])
    model = model_class(config, dataset_list)
    return model

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def normal_initialization(module, initial_range=0.02):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def get_parameter_list(model, detach=True):
    para_list = [p.detach().cpu() for p in model.parameters()]
    return para_list

def flatten_state_dict(state_dict):
    return torch.cat([p.flatten() for _, p in state_dict.items()])

def load_config(config : dict):
    path = os.path.join('configs', config['dataset'].lower() + '.yaml')
    with open(path, "r") as stream:
        config['data'] = yaml.safe_load(stream)
    config['data']['dataset'] = deepcopy(config['dataset'])
    config.pop('dataset')
    model_name = deepcopy(config['model'])
    path = os.path.join('configs', 'basemodel.yaml')
    with open(path, "r") as stream:
        config.update(yaml.safe_load(stream))
    path = os.path.join('configs', model_name.lower() + '.yaml')
    with open(path, "r") as stream:
        model_config = yaml.safe_load(stream)
        for key, value in model_config.items():
            config[key].update(value)
    config['model']['model'] = model_name
    return config

def load_sweep_config(config):
    path = os.path.join('sweep', config['model']['model'].lower() + '.yaml')
    sweep_config = {}
    with open(path, "r") as stream:
        model_config = yaml.safe_load(stream)
        for key, value in model_config.items():
            sweep_config[key] = value
    return sweep_config

def transform_config_into_sweep_config(sweep_config, config):
    for category_k, category_v in config.items():
        for entry_k, entry_v in category_v.items():
            if sweep_config['parameters'].get(category_k + '.' + entry_k, None) == None:
                sweep_config['parameters'][category_k + '.' + entry_k] = {'value': entry_v}
    return sweep_config

def transform_sweep_config_into_config(sweep_config):
    config = {'data': {}, 'model': {}, 'train': {}, 'eval': {}}
    for k, v in sweep_config.items():
        key = k.split('.')
        config[key[0]][key[1]] = v
    return config


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

def get_optimal_value(effect):
    epoch = effect['test_auc'].index(max(effect['test_auc']))
    optimal_effect = {'epoch': epoch+1,
                      'train_acc': effect['train_acc'][epoch],
                      'train_auc': effect['train_auc'][epoch],
                      'val_acc': effect['val_acc'][epoch],
                      'val_auc': effect['val_auc'][epoch],
                      'test_acc': effect['test_acc'][epoch],
                      'test_auc': effect['test_auc'][epoch]}

    return optimal_effect

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, hard=False, eps=1e-10):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.eps = eps

    def forward(self, scores, tau=1):
        device = scores.device
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores, device=device)
        onehot_approx = torch.zeros_like(scores, device=device)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([self.eps], device=device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot, device=device)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

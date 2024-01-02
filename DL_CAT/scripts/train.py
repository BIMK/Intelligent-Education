import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
sys.path.append('..')
import pyat
import random
from sklearn.utils import shuffle


dataset = 'assistment'
rcd_model = 'irt'

if rcd_model == 'irt':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)


# read datasets
train_set = (pd.read_csv(f'../datasets/{dataset}/train_triplets.csv', encoding='utf-8'))
train_triplets = train_set.head(int(len(train_set) * 1)).to_records(index=False)

concept_map = json.load(open(f'../datasets/{dataset}/concept_map.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
metadata = json.load(open(f'../datasets/{dataset}/metadata.json', 'r'))

# construct
train_data = pyat.TrainDataset(train_triplets, concept_map,
                               metadata['num_train_students'], metadata['num_questions'], metadata['num_concepts'])

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname)s] %(message)s',
)

if rcd_model == 'irt':
    # 参数配置
    config = {
        'learning_rate': 0.002,
        'batch_size': 32,
        'num_epochs': 20,
        'num_dim': 1,
        'device': 'cpu',
        'model_after_param': '../models/irt/' + dataset + '_test.pt',
        'model_start_param': '../models/irt/' + dataset + 'start_param.pt',
        'model_save_path': '../models/irt/' + dataset + '.pt',
        'model_save_final_path': '../models/irt/' + dataset + '_final.pt',
        'dnn_best_save_path': '../models/irt/' + dataset + '_best_dnn.pt',
        'dnn_final_save_path': '../models/irt/' + dataset + '_final_dnn.pt'
    }
    model = pyat.IRTModel(**config)

model.adaptest_init(train_data)
start_theta = model.adaptest_train(train_data)
model.adaptest_save(model.config['model_save_path'])

if rcd_model == 'irt':
    model1 = pyat.IRTModel(**config)
    model1.adaptest_init(train_data)
    model1.adaptrain_preload(model.config['model_save_path'], start_theta)
    model1.start_save(model.config['model_start_param'])
    model1.meta_pre(train_data)






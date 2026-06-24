import sys
import json
import logging
import numpy as np
import pandas as pd
import torch

sys.path.append('..')
import pyat
import diffusion

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

dataset = 'Python'
step = 'test'
num = '20'
rcd_model = 'ncd'
# read datasets
train_triplets = pd.read_csv(f'../datasets/PTADisc/{dataset}/{step}_{num}_sort.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../datasets/PTADisc/{dataset}/concept_map_{dataset}.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
dataInfo = json.load(open(f'../datasets/PTADisc/{dataset}/info.json', 'r'))

print(dataInfo[f'{step}_cnt'])
print(dataInfo['problem_cnt'])
# construct
train_data = pyat.TrainDataset(train_triplets, concept_map,
                               dataInfo[f'{step}_cnt'], dataInfo['problem_cnt'], dataInfo['concept_cnt'])

train_set = (pd.read_csv(f'../datasets/PTADisc/{dataset}/{step}_{num}_sort.csv', encoding='utf-8'))
train_triplets = train_set.head(int(len(train_set) * 1)).to_records(index=False)
val_set = train_set.tail(int(len(train_set) * 0.1)).to_records(index=False)
val_data = pyat.TrainDataset(val_set, concept_map,
                             dataInfo[f'{step}_cnt'], dataInfo['problem_cnt'], dataInfo['concept_cnt'])
if rcd_model == 'irt':
    config = {
        'learning_rate': 0.002,
        'batch_size': 32,
        'num_epochs': 20,
        'num_dim': 1,
        'device': 'cpu',
        'model_after_param': '../models/irt/' + dataset + '_test.pt',
        'model_start_param': '../models/irt/' + dataset + 'start_param.pt',
        'model_save_path': '../models/irt/' + dataset + '_80_train.pt',
        'model_save_final_path': '../models/irt/' + dataset + '_final.pt',
        'dnn_best_save_path': '../models/irt/' + dataset + '_best_dnn.pt',
        'dnn_final_save_path': '../models/irt/' + dataset + '_final_dnn.pt'
    }
    # model = pyat.IRTModel(**config)

elif rcd_model == 'mirt':
    config = {
        'learning_rate': 0.002,
        'batch_size': 32,
        'num_epochs': 10,
        'num_dim': 5,
        'device': 'cpu',
        'model_start_param': '../models/mirt/start_param.pt',
        'model_save_path': '../models/mirt/' + dataset + '.pt',
        'model_save_final_path': '../models/mirt/' + dataset + '_final.pt',
        'dnn_best_save_path': '../models/mirt/' + dataset + '_best_dnn.pt',
        'dnn_final_save_path': '../models/mirt/' + dataset + '_final_dnn.pt'
    }
    model = pyat.IRTModel(**config)


elif rcd_model == 'ncd':
    config = {
        'learning_rate': 0.002,
        'batch_size': 32,
        'num_epochs': 10,
        'device': 'cuda',
        'model_start_param': '../models/ncd/start_param.pt',
        'model_save_path': '../models/ncd/' + dataset + f'_80_{step}.pt',
        'model_save_final_path': '../models/ncd/' + dataset + '_final.pt',
        'dnn_save_path': '../models/ncd/' + dataset + '_dnn.pt'
    }
    # model = pyat.NCDModel(**config)

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname)s] %(message)s',
)



if rcd_model == 'irt':
    print("Model = irt")
    model = pyat.IRTModel(**config)  
    model.adaptest_init(train_data)  
    model.adaptest_train(train_data)  

    data = [] 
    # for i in range(0, dataInfo['student_cnt']):  
    #     value = model.get_theta(i)
    #     
    #     student_data = {
    #         "stu_id": i,
    #         "theta": value.tolist()
    #     }
    #     data.append(student_data) 

    for i in range(0, dataInfo[f'{step}_cnt']): 
        value = -1 * model.get_theta(i)
        data.append(value.tolist())

   
    with open(f'../datasets/PTADisc/IRT/{dataset}/{dataset}_stu_{step}_theta_unnum.json', 'w') as file:
        json.dump(data, file, indent=4)  
    # model.adaptest_save(f'../models/irt/{dataset}_80_train.pt')  



elif rcd_model == 'ncd':
    print("Model = ncd")
    model = pyat.NCDModel(**config)
    model.adaptest_init(train_data)
    start_theta = model.adaptest_train(train_data, val_data)
    # model.adaptrain_preload(model.config['model_save_path'], start_theta)
    # model.start_save(model.config['model_start_param'])
    stu_theta = start_theta.cpu().detach().numpy().tolist()
    with open(f'../datasets/PTADisc/NCD+MAAT/{dataset}/{dataset}_stu_{step}_theta_unnum.json', 'w') as f:
        json.dump(stu_theta, f, indent=4)


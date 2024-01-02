import sys
import json
import datetime
import logging
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('..')
import pyat
from pyat.model.DNN_train import Net_loss, Net_loss_ncd, Net_loss_ncd2

dataset = 'assistment'
rcd_model = 'irt'

if rcd_model == 'irt':
    sels = ['random', 'fisher', 'Maat', 'Maat_cov', 'DLCAT']

    config = {
        'learning_rate': 0.0025,
        'batch_size': 32,
        'num_epochs': 8,
        'num_dim': 1,
        'device': 'cpu',
        'model_after_param': '../models/irt/' + dataset + 'after.pt',
        'model_save_path': '../models/irt/' + dataset + '.pt',
        'dnn_save_path': '../models/irt/' + dataset + '_best_dnn.pt'
    }


for sel in sels:
    # read datasets
    test_triplets = pd.read_csv(f'../datasets/{dataset}/test_triplets.csv', encoding='utf-8').to_records(index=False)
    concept_map = json.load(open(f'../datasets/{dataset}/concept_map.json', 'r'))
    concept_map = {int(k): v for k, v in concept_map.items()}
    metadata = json.load(open(f'../datasets/{dataset}/metadata.json', 'r'))

    test_data = pyat.AdapTestDataset(test_triplets, concept_map,
                                     metadata['num_test_students'], metadata['num_questions'], metadata['num_concepts'])

    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s %(levelname)s] %(message)s',
    )

    test_length = 20
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    strategies = pyat.MAATStrategy(n_candidates=10)
    if rcd_model == 'irt':
        seed = 4
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = pyat.IRTModel(**config)
        model.adaptest_init(test_data)
        model.adaptest_preload(model.config['model_save_path'])
        # Loading model
        num = 2 * (model.config['num_dim']) + 1
        net_loss = Net_loss(num)
        net_loss.load_snapshot(net_loss, model.config['dnn_save_path'])


    test_data.reset()
    ran_auc_list, fisher_auc_list, Maat_auc_list, Maat_Cov_auc_list, DLCAT_auc_list, DLCAT_Cov_auc_list, ran_acc_list, fisher_acc_list, Maat_Cov_list, Maat_Cov_acc_list, DLCAT_acc_list, real_auc_list, real_acc_list = pyat.AdapTestDriver.run(
        model, net_loss, strategies, test_data, test_length, f'../results/{now}', sel, rcd_model)

plt.figure()
plt.plot(range(len(ran_auc_list)), ran_auc_list, color='red', linewidth=2.0,  label='random')
plt.plot(range(len(fisher_auc_list)), fisher_auc_list, color='blue', linewidth=2.0, label='fisher')
plt.plot(range(len(Maat_auc_list)), Maat_auc_list, color='green', linewidth=2.0, label='Maat')
plt.plot(range(len(Maat_Cov_auc_list)), Maat_Cov_auc_list, color='pink', linewidth=2.0, label='Maat_Cov')
plt.plot(range(len(DLCAT_auc_list)), DLCAT_auc_list, color='black', linewidth=2.0, label='DLCAT')

plt.legend()
plt.title(f'Selected_Strategy/{rcd_model}{-num}{-seed}/auc')
plt.show()

time.sleep(8)

plt.figure()
plt.plot(range(len(ran_acc_list)), ran_acc_list, color='red', linewidth=2.0,  label='random')
plt.plot(range(len(fisher_acc_list)), fisher_acc_list, color='blue', linewidth=2.0, label='fisher')
plt.plot(range(len(Maat_Cov_list)), Maat_Cov_list, color='green', linewidth=2.0, label='Maat')
plt.plot(range(len(Maat_Cov_acc_list)),  Maat_Cov_acc_list, color='pink', linewidth=2.0, label='Maat_Cov')
plt.plot(range(len(DLCAT_acc_list)), DLCAT_acc_list, color='black', linewidth=2.0, label='DLCAT')

plt.legend()
plt.title(f'Selected_Strategy/{rcd_model}{-num}{-seed}/acc')
plt.show()
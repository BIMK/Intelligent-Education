import logging

import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from NASCD import NASCD
import random
import torch.backends.cudnn as cudnn

DECSPACE = []

DECSPACE.append([0,0,10,  2,3,11 , 0,0,9,  5,4,10,  1,6,10,  1,7,10,  8,0,6 , 9,0,13])


from sklearn.utils import shuffle
from tqdm import tqdm
import os
import numpy as np
def split_dataset(train_data,valid_data,test_data,split=0.5):
    train_data = pd.concat([train_data,valid_data,test_data])
    train_set,test_set = pd.DataFrame(),pd.DataFrame()
    for index, value in enumerate(tqdm(train_data.groupby('user_id'))):
        train_size = int(len(value[1]) * split)+1

        item = value[1]
        item = shuffle(item)

        train_set = pd.concat([train_set,item[:train_size]])
        test_set = pd.concat([test_set,item[train_size:]])

    train_set = train_set.reset_index()
    test_set = test_set.reset_index()
    return train_set,test_set,test_set






batch_size = 128


# for names in ['ASSIST','SLP']:
for names in ['slp']:

    # list_ratio = [0.5,0.6,0.7,0.8] if names=='SLP' else [0.6,0.7,0.8]

    list_ratio =[0.8]
    for split_ratio in list_ratio:
        print(split_ratio)
        from utils_dataset_train import  get_dataset
        # train_set, valid_set, test_set ,user_n,item_n,knowledge_n = get_dataset(name='slp',ratio= split_ratio)
        train_set, test_set ,user_n,item_n,knowledge_n = get_dataset(name='slp',ratio= split_ratio)



        for archi_i, NASDEC in enumerate(DECSPACE):

            result_1 =[]
            result_2 =[]

            for run_i in range(10):

                logging.getLogger().setLevel(logging.INFO)
                cdm = NASCD(knowledge_n, item_n, user_n,dec=NASDEC)

                best_result, last_result =cdm.train(train_set, test_set, epoch=50, device="cuda",lr=0.002) # SLP epoch 20: 0.8442

                del cdm
                result_1.append(best_result)
                result_2.append(last_result)


            root_path_1 ='experiment/NAS_GCD_archi_{}_{}_{}_best_result_.txt'.format(archi_i,names,split_ratio)
            root_path_2 ='experiment/NAS_GCD_archi_{}_{}_{}_last_result_.txt'.format(archi_i,names,split_ratio)

            result_1 = np.array(result_1)
            result_2 = np.array(result_2)
            result_1 = np.vstack([result_1, np.mean(result_1,axis=0 )])
            result_2 = np.vstack([result_2, np.mean(result_2,axis=0 )])

            np.savetxt(root_path_1,np.array(result_1),delimiter=' ')
            np.savetxt(root_path_2,np.array(result_2),delimiter=' ')





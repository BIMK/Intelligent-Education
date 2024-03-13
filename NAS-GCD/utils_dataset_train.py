import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

import json



def transform(user, item, item2knowledge, score, batch_size,knowledge_n):

    # knowledge_emb1 = torch.zeros((len(item), knowledge_n))
    knowledge_emb = []
    for idx in range(len(item)):
        # knowledge_emb1[idx, item2knowledge[ item[idx]-1 ]==1 ] = 1  # error
        knowledge_emb.append(item2knowledge[ item[idx]-1 ] )

    knowledge_emb = np.array(knowledge_emb)
    knowledge_emb = torch.Tensor(knowledge_emb)


    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def get_dataset(name = 'assist12',batch_size = 32,ratio = 0.5):

    if ratio == 0.8:
        # with open("./data/数据集/数据集/data/{}/{}_val_set.json".format(name,name), encoding='utf8') as i1_f:
        #     valid_data1 = json.load(i1_f)

        with open("./data/数据集/数据集/data/{}/{}_train_set_8.json".format(name,name), encoding='utf8') as i_f:
            train_data1 = json.load(i_f)

        with open("./data/数据集/数据集/data/{}/{}_test_set_8.json".format(name,name), encoding='utf8') as i2_f:
            test_data1 = json.load(i2_f)
    else:
        ratio = int(10*ratio)
        with open("../数据集/数据集/data/{}/{}_val_set_{}.json".format(name,name,ratio), encoding='utf8') as i1_f:
            valid_data1 = json.load(i1_f)

        with open("../数据集/数据集/data/{}/{}_train_set_{}.json".format(name,name,ratio), encoding='utf8') as i_f:
            train_data1 = json.load(i_f)

        with open("../数据集/数据集/data/{}/{}_test_set_{}.json".format(name,name,ratio), encoding='utf8') as i2_f:
            test_data1 = json.load(i2_f)



    item2knowledge = np.loadtxt("./data/数据集/数据集/data/{}/{}_item2knowledge.txt".format(name,name))



    train_data = {}
    usr = []
    exer=[]
    score = []
    for item in train_data1:
        usr.append(item['user_id'])
        exer.append(item['exer_id'])
        score.append(item['score'])
    # for item in valid_data1:
    #     usr.append(item['user_id'])
    #     exer.append(item['exer_id'])
    #     score.append(item['score'])
    train_data['user_id'] = usr
    train_data['exer_id'] = exer
    train_data['score'] = score




    # valid_data = {}
    # usr = []
    # exer=[]
    # score = []
    # for item in valid_data1:
    #     usr.append(item['user_id'])
    #     exer.append(item['exer_id'])
    #     score.append(item['score'])
    # valid_data['user_id'] = usr
    # valid_data['exer_id'] = exer
    # valid_data['score'] = score


    test_data = {}
    usr = []
    exer=[]
    score = []
    for item in test_data1:
        usr.append(item['user_id'])
        exer.append(item['exer_id'])
        score.append(item['score'])
    test_data['user_id'] = usr
    test_data['exer_id'] = exer
    test_data['score'] = score

    user_n = np.max(train_data['user_id'])
    # exer_n = np.max([np.max(train_data['exer_id']), np.max(valid_data['exer_id']), np.max(test_data['exer_id'])])
    exer_n = np.max([np.max(train_data['exer_id']), np.max(test_data['exer_id'])])
    knowledge_n = item2knowledge.shape[1]

    # train_set, valid_set, test_set = [
    #     transform(data["user_id"], data["exer_id"], item2knowledge, data["score"], batch_size,knowledge_n)
    #     for data in [train_data, valid_data, test_data]
    # ]
    train_set, test_set = [
        transform(data["user_id"], data["exer_id"], item2knowledge, data["score"], batch_size,knowledge_n)
        for data in [train_data, test_data]
    ]
    return  train_set,  test_set, user_n,exer_n,knowledge_n

    # return  train_set, valid_set, test_set, user_n,exer_n,knowledge_n

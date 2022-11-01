"""
此文件用来评价预测模型的多目标指标
"""
import pandas as pd
import numpy as np
import heapq
import json
from tqdm import tqdm
import dask.dataframe as dd

from evluate import *
from utils import reduce_mem
# from calculate_diversity import cal_div
# 'yelp2018''gowalla''pinterest'

dataset = 'Music'
recommender = 'LightGCN'
topk = 100
if dataset == 'anime':
    maxdiv = 82
    st = ','
elif dataset == 'ml-10m':
    maxdiv = 20
    st = '|'
elif dataset == 'Music':
    maxdiv = 404
    st = ','

def minmax(List):
    return (np.mean(List)-min(List))/(max(List)-min(List))
# sum(list(all_rating[3].values())[0:10])/10
all_rating = np.load('util_data/' + recommender + dataset + '_rating.npy', allow_pickle=True).item()
data = pd.read_csv('data/' + dataset + '.train', sep=',', header=None, usecols=[0, 1], names=["user", "item"])
test = pd.read_csv('data/' + dataset + '.test', sep=',', header=None, usecols=[0, 1], names=["user", "item"])
div_list = pd.read_csv('util_data/' + dataset + '_div.csv', sep=',', header=None, usecols=[0, 1], names=["item", "genre"])
div_list = div_list.set_index("item")
# Si = cal_div(dataset)
# df = pd.read_csv('util_data/' + dataset + '_Si.csv', sep=',')
# df = df.compute()
# df = reduce_mem(df)

# # 把VALUES列用eval抓换就好了
# df['VALUES'] = df.apply(lambda x: eval(x.VALUES), axis=1)
count = 0
Pre, Nov, Div, Hit = [], [], [], []
except_user = []
nov_list = data["item"].value_counts()
buy_record = data.groupby('user')['item'].apply(list)
truth_record = test.groupby('user')['item'].apply(list)
length = 10

for key, value in tqdm(all_rating.items()):
    # if key == 19:
        items = []
        rank = heapq.nlargest(topk, value.items(), lambda x: x[1])
        for item, _ in rank:
            if len(items) == length:
                break
            if item in buy_record[key]:
                continue
            items.append(item)
        if len(items) != length:
            except_user.append(key)
            continue
        try:
            Pre.append(precision(items, truth_record[key]))
            Hit.append(hit(items, truth_record[key]))
            Nov.append(novelty(items, nov_list))
            Div.append(diversity(items, div_list, maxdiv, st))
        except:
            count += 1



# print('precisioin=%.4f\tnovelty=%.1f\tdiversity=%.4f' % (np.mean(Pre), np.mean(Nov), np.mean(Div)))
# print('precisioin=%.4f\tnovelty=%.4f\t' % (np.mean(Pre), 1 - np.mean(Nov)/nov_list.max()))
print('precisioin=%.4f\tnovelty=%.4f\tdiversity=%.4f\tHit=%.4f\t' % (np.mean(Pre), 1 - np.mean(Nov), np.mean(Div), np.mean(Hit)))
print(count)
print('novelty=%.4f' % (min(Nov)))


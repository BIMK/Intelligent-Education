from sklearn.cluster import KMeans
import numpy as np
import torch
from tqdm import trange
from sklearn.cluster import AgglomerativeClustering
import scipy.sparse as sp
from sklearn.cluster import spectral_clustering, KMeans, Birch
import heapq

# def cluster(user_cluster_num, user_num, item_num, dataset):
# 	model = torch.load(dataset + '.model')
# 	U, I, u_text, i_text = [], [], [], []
# 	for i in range(user_num):
# 		U.append(i)
# 	Pu = model.get_embeding(user_id=torch.LongTensor(U).cuda(), item_id=None, type='user')
#
# 	# for i in range(item_num):
# 	#     I.append(i)
# 	# Pi = model.get_embeding(item_id=torch.LongTensor(I).cuda(),  user_id=None, type='item')
# 	Pu = Pu.cpu().detach().numpy()
# 	# Pi = Pi.cpu().detach().numpy()
#
# 	# K-Means
# 	user_estimator = KMeans(n_clusters=user_cluster_num)
# 	user_estimator.fit(Pu)
# 	Cluster_results = user_estimator.labels_  # array
# 	return Cluster_results

def cluster(Data, rating, n_clusters):
    tempnum = 1000
    count = 0
    topk = 100#用每个用户相似的前100个做相似性计算
    buy_record = Data.buy_record
    num = Data.user_num
    W = np.zeros([tempnum,tempnum],dtype = np.int32)
    rec = {}
    for key, value in rating.items():
        count+=1
        rec[key] = []
        rank = heapq.nlargest(100, value.items(), lambda x: x[1])
        for item, _ in rank:
            if len(rec[key]) == topk:
                break
            if item in buy_record[key]:
                continue
            rec[key].append(item)
        if count == tempnum:
            break


    for i in range(tempnum):
        for j in range(tempnum):
            if i == j:
                continue
            W[i][j] = len(set(rec[i]) & set(rec[j]))
    # mask = W.astype(bool)
    # graph = sp.coo_matrix(W)
    Cluster_results = Birch(n_clusters=100).fit(-W).labels_
    # Cluster_results = KMeans(n_clusters=100).fit(-W).labels_
    # Cluster_results = spectral_clustering(graph, n_clusters=100, assign_labels='kmeans')


    for i in range(100):
        tmp = np.where(Cluster_results == i)[0]
        W_new = np.zeros([len(tmp),len(tmp)],dtype = np.int32)
        a1=-1
        for a in tmp:
            a1+=1
            b1 = -1
            for b in tmp:
                b1+=1
                if a == b:
                    continue
                W_new[a1][b1] = len(set(rec[a]) & set(rec[b]))
    return Cluster_results


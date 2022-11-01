from tqdm import trange
import numpy as np
import time
import json
import warnings


from MFEA.mfea import mfea
from BENCHMARK.task import Task, generate_tasks
from data_laoder import Dataset
from evluate import *
from cluster import cluster




if __name__=="__main__":
    dataset = 'ml-10m'
    recommender = 'MF'
    # Si = json.load(open(dataset + '_Si.txt'))
    Data = Dataset('data/' + dataset +'.train')
    # Data.Si = json.load(open('util_data/' + dataset + '_Si.txt'))
    user_cluster_num = Data.user_num//10
    # for i in range(user_cluster_num):
    #     print("label %d user num is %d\n" % (i, len(list(np.where(Cluster_results == i)[0]))))
    rating = np.load('util_data/' + recommender + dataset + '_rating.npy', allow_pickle=True).item()
    print('Strat Cluser')
    Cluster_results = cluster(Data, rating, user_cluster_num)
    M = 2
    Pre_pre, Pre_nov, Pre_len, Pre_div, Ave_nov, Ave_div = [], [], {}, [], [], []


    warnings.simplefilter('error', RuntimeWarning)
    t1 = time.time()
    for i in trange(user_cluster_num):
        user = list(np.where(Cluster_results == i)[0])
        tasks = generate_tasks(M, user, rating)
        NDset, Log, fin_obj = mfea(tasks, user, Data)
        pre_pre, pre_nov, pre_len, excat_user, pre_div, ave_nov, ave_div = evluate(NDset, matrix_test, matrix_train, item_interact_frequency, user, Si,ratings, fin_obj)
        Pre_pre.append(pre_pre)
        Pre_nov.append(pre_nov)
        Pre_div.append(pre_div)
        Ave_div.append(ave_div)
        Ave_nov.append(ave_nov)
        j = 0
        for u in user:
            if u in excat_user:
                continue
            Pre_len[int(u)] = pre_len[j]
            j = j+1

        # Nov_nov.append(nov_nov)
        # Nov_pre.append(nov_pre)
        # Nov_len.append(nov_len)


    print(str(M) +'\t' + dataset + '\tprecisioin=%.4f\tnovelty=%.1f\tdiversity=%.2f\tave_nov=%.1f\tave_div=%.2f\ttime=%.1f\t' % (np.mean(Pre_pre), np.mean(Pre_nov), 1/np.mean(Pre_div), np.mean(Ave_nov), 1/np.mean(Ave_div), time.time()-t1))
    f = open('evaluate_result_store', 'a')
    f.write(str(M) +'\t' + dataset + '\tprecisioin=%.4f\tnovelty=%.1f\tdiversity=%.2f\tave_nov=%.1f\tave_div=%.2f\ttime=%.1f\t' % (np.mean(Pre_pre), np.mean(Pre_nov), 1/np.mean(Pre_div), np.mean(Ave_nov), 1/np.mean(Ave_div), time.time()-t1) + '\n')
    f.close()
    json.dump(Pre_len, open(dataset + "_len.txt", 'w'))
    json.dump(Log, open(dataset + "_old_HV.txt", 'w'))





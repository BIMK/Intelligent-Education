import numpy as np
from sklearn.preprocessing import normalize


def Diversity(X, Si):
    #可能会出现长度为1的情况
    if len(X) == 1 or 0:
        return 1
    diversity = 0.0
    for i in X:
        for j in X:
            if i == j:
                continue
            try:
                diversity += Si[str(i-1)][str(j-1)]
            except:
                continue
    diversity = diversity / (len(X) * (len(X) - 1))
    return diversity


class Task(object):
    def __init__(self, M, rating):
        self.M = M
        self.rating = rating



    def fnc(self, dec, Mi, Si):
        obj = np.zeros([self.M])
        if self.M == 2:
            try:
                obj[0] = np.sum(self.rating[i] for i in dec)/len(dec)
                obj[1] = np.sum(Mi[i] for i in dec)/len(dec)

            except:
                pass
        if self.M == 3:
            try:
                obj[0] = np.sum(self.rating[i] for i in dec) / len(dec)
                obj[1] = np.sum(Mi[i] for i in dec) / len(dec)
                obj[2] = Diversity(dec, Si)
            except:
                return obj
        return obj


def generate_tasks(M, User, rating):
    tasks = []

    # '''
    # 取0号用户为样本，9个最相似的人为任务
    # '''
    # k_user = list(np.argsort(W[0])[::-1][0:9])
    # k_user.append(0)
    for user in User:
        tasks.append(Task(M, rating[user]))

    # rating = rating[User]
    # rating = normalize(rating, axis=1, norm='max')
    # for i in User:
    #     for j in User:
    #         if i == j:
    #             continue
    #         W[i][j] = np.sum(abs(rating[i] - rating[j]))
    return tasks

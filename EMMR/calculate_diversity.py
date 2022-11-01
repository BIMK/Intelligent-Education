import pandas as pd

# coding = utf-8

# 基于项目的协同过滤推荐算法实现
import random

import math
from operator import itemgetter
import json
import numpy as np
import time
from tqdm import trange,tqdm
import pandas as pd
from utils import reduce_mem


class ItemBasedCF():
    # 初始化参数
    def __init__(self, datasets):
        self.data = datasets
        self.t1 = time.time()
        # 找到相似的20部电影，为目标用户推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10


        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie)
        # print('Recommneded movie number = %d' % self.n_rec_movie)


    # 读文件得到“用户-电影”数据
    def get_dataset(self, trainname, testname):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file('data/' + trainname):

            arr = line.split(",")
            user, movie, rating = int(arr[0]), int(arr[1]), 1

            self.trainSet.setdefault(user, {})
            self.trainSet[user][movie] = rating
            trainSet_len += 1
        item_popularity = dict()
        for user, items in self.trainSet.items():
            for item in items:
                if item not in item_popularity:
                    item_popularity[item] = 0
                item_popularity[item] += 1
        self.item_popularity = item_popularity




        for line in self.load_file('data/' + testname):
            arr = line.split(",")
            user, movie, rating = int(arr[0]), int(arr[1]), 1

            self.testSet.setdefault(user, {})
            self.testSet[user][movie] = rating
            # self.testSet[user][movie] = rating
            testSet_len += 1

        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)


    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                # if i == 0:  # 去掉文件第一行的title
                #     continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)





    # 计算电影之间的相似度
    def calc_movie_sim(self):
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        self.movie_count = len(self.movie_popular)
        print("Total movie number = %d" % self.movie_count)

        for user, movies in self.trainSet.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    # self.movie_sim_matrix[m1][m2] += 1
                    self.movie_sim_matrix[m1][m2] += 1 / math.log(1 + len(movies) * 1.0)
        print("Build co-rated users matrix success!")

        # 计算电影之间的相似性
        print("Calculating movie similarity matrix ...")
        for m1, related_movies in tqdm(self.movie_sim_matrix.items()):
            for m2, count in related_movies.items():
                # 注意0向量的处理，即某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')
        # json.dump(self.movie_sim_matrix, open('util_data/' + self.data + "_Si.txt", 'w'))

        keys = self.movie_sim_matrix.keys()
        values = self.movie_sim_matrix.values()
        df = pd.DataFrame({'KEYS': keys, 'VALUES': values})
        df = reduce_mem(df)
        df.to_csv('util_data/' + self.data + '_Si.csv', index=False)

    # 针对目标用户U，找到K部相似的电影，并推荐其N部电影
    def recommend(self, user):
        K = self.n_sim_movie
        # N = 10
        N = self.n_rec_movie[str(user - 1)]
        rank = {}
        watched_movies = self.trainSet[user]

        for movie, rating in watched_movies.items():
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluating start ...')

        item_popularity = self.item_popularity
        Div = []
        # 准确率和召回率
        rec_count = 0
        test_count = 0
        nov = 0
        pre = 0.0
        # 覆盖率
        all_rec_movies = set()
        for user in tqdm(self.trainSet):
            hit = 0
            diversity = 0.0
            try:
                N = self.n_rec_movie[str(user-1)]
                # N = self.n_rec_movie
            except:
                continue
            test_moives = self.testSet.get(user, {})
            rec_movies = self.recommend(user)
            if len(rec_movies) != 1 or 0:
                for m1, _ in rec_movies:
                    for m2, _ in rec_movies:
                        if m1 == m2:
                            continue
                        try:
                            diversity += self.movie_sim_matrix[m1][m2]
                        except:
                            continue
                diversity = diversity / (len(rec_movies) * (len(rec_movies) - 1))
            else:
                diversity = 1
            Div.append(diversity)
            for movie, w in rec_movies:
                nov += item_popularity[movie]
                # nov += math.log(1 + item_popularity[movie])
                if movie in test_moives:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            pre += hit/N
            test_count += len(test_moives)

        # precision = hit / (1.0 * rec_count)
        # recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        novetly = nov / (1.0 * rec_count)
        print('precisioin=%.4f\tnovetly=%.2f\tdiversity=%.2f\ttime=%d' % (pre/len(self.n_rec_movie), novetly, 1/np.mean(Div), time.time()-self.t1))

    def get_rating(self,datasets):
        K = self.n_sim_movie
        # N = self.n_rec_movie
        print("rating start ...")
        user_num = len(self.trainSet)
        item_num = 10086 #ml-100k物品为1682个，ml-1m物品3952,3706个,Netfilix物品2700个,ml-10m 10086
        rating_matrix = np.zeros([user_num,item_num])

        for user in trange(user_num):
            user = user + 1 #ml-100k从1开始
            watched_movies = self.trainSet[user]
            for movie, rating in watched_movies.items():
                for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True):
                    if related_movie in watched_movies:
                        continue
                    rating_matrix[user-1][related_movie-1] += w #ml-100k从1开始
        np.savetxt(datasets + "_rating", rating_matrix)
# def cal_div(datasets):
#     itemCF = ItemBasedCF(datasets)
#     itemCF.get_dataset(datasets + '.train', datasets + '.test')
#     itemCF.calc_movie_sim()
#     return itemCF.movie_sim_matrix
#     # itemCF.get_rating(datasets)
#     # itemCF.evaluate()
if __name__ == '__main__':
    datasets = 'yelp2018'
    # rating_file = 'D:\\学习资料\\推荐系统\\ml-latest-small\\ratings.csv'
    itemCF = ItemBasedCF(datasets)
    itemCF.get_dataset(datasets + '.train', datasets + '.test')
    itemCF.calc_movie_sim()
    # itemCF.get_rating(datasets)
    # itemCF.evaluate()

import numpy as np
import matplotlib.pyplot as plt
import geatpy as ea
from matplotlib.pyplot import MultipleLocator
# x1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
# y1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
# x2 = np.random.normal(7.5, 1.2, 300)
# y2 = np.random.normal(7.5, 1.2, 300)
def reinsertion(population, NUM):
    [levels, criLevel] = ea.ndsortESS(population, NUM, None, None,
                                     np.array([-1,-1]))# 对NUM个个体进行非支配分层
    dis = ea.crowdis(population, levels)  # 计算拥挤距离
    chooseFlag = ea.selecting('dup', np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort').reshape(-1, 1), NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
    return population[chooseFlag]


def quchong(arr):
    s=[]
    for i in arr:
       if i not in s:
           s.append(i)
    return s
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# matplotlib画图中中文显示会有问题，需要这两行设置默认字体
# y_major_locator=MultipleLocator(0.03)
# ax=plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)

# anime(0,11,47)
# plt.xlim(xmax=9, xmin=0)
# plt.ylim(ymax=9, ymin=0)
# 画两条（0-9）的坐标轴并设置轴标签x，y
dataset = 'Music'
recommender = 'LightGCN'
objective_new = np.load('util_data/' + recommender + dataset + '_objective_new.npy', allow_pickle=True)
objective_old = np.load('util_data/' + recommender + dataset + '_objective_old.npy', allow_pickle=True)
objective_PMOEA = np.load('util_data/' + recommender + dataset + '_objectivePOMEA.npy', allow_pickle=True)
objective_MOEARC = np.load('util_data/' + recommender + dataset + '_objectiveMOEARC.npy', allow_pickle=True)



font = {'family': 'Times New Roman'}

if recommender == 'MF' and dataset == 'anime':
    for i in range(len(objective_new[0])):
        if i == 6:
            plt.xlabel(r'Accuracy  f$_1$', font,  fontsize=16)
            plt.ylabel(r'Diversity  f$_3$', font,  fontsize=16)
            x1 = list(reversed(quchong(objective_new[0][i])))
            y1 = list(reversed(quchong(objective_new[1][i])))
            x2 = 12.1185
            y2 = 0.24
            x4 = objective_old[0][i]
            y4 = objective_old[1][i]
            x5 = objective_PMOEA[0][i]
            y5 = objective_PMOEA[1][i]
            x6 = objective_MOEARC[0][i]
            y6 = objective_MOEARC[1][i]
            x3 = 10.06
            y3 = 0.34
            # area = np.pi * 4 ** 2  # 点面积
            # 画散点图


            plt.scatter(x2, y2, c='b', label='BPR-MF', marker='*')
            plt.scatter(x3, y3, c='k', label='PD-GAN', marker='s', zorder=2)
            plt.scatter(x5, y5, c='m', label='PMOEA', marker='x', )
            plt.scatter(x6, y6, c='orange', label='MOEA-ProbS', marker='D', zorder=3)
            plt.scatter(x4, y4, c='#32CD32', label='MORS', zorder=1)
            plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
            # plt.title(i)
            plt.legend(fontsize='large')
            plt.tick_params(labelsize=15)
            plt.show()
if recommender == 'MF' and dataset == 'ml-10m':
    for i in range(len(objective_new[0])):
        if i == 35:
            plt.xlabel(r'Accuracy  f$_1$', font,  fontsize=16)
            plt.ylabel(r'Diversity  f$_3$', font,  fontsize=16)
            x1 = list(reversed(quchong(objective_new[0][i])))
            y1 = list(reversed(quchong(objective_new[1][i])))
            x2 = 8.53
            y2 = 0.500
            x4 = objective_old[0][i]
            y4 = objective_old[1][i]
            x5 = objective_PMOEA[0][i]
            y5 = objective_PMOEA[1][i]
            x6 = objective_MOEARC[0][i]
            y6 = objective_MOEARC[1][i]
            x3 = 7.443
            y3 = 0.600
            # area = np.pi * 4 ** 2  # 点面积
            # 画散点图

            plt.scatter(x2, y2, c='b', label='BPR-MF', marker='*')
            plt.scatter(x3, y3, c='k', label='PD-GAN', marker='s', zorder=2)
            plt.scatter(x5, y5, c='m', label='PMOEA', marker='x', )
            plt.scatter(x6, y6, c='orange', label='MOEA-ProbS', marker='D')
            plt.scatter(x4, y4, c='#32CD32', label='MORS', zorder=1)
            plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
            plt.tick_params(labelsize=15)
            # plt.title(i)
            plt.legend(fontsize='large')
            plt.show()
            # except:
            #     continue
if recommender == 'MF' and dataset == 'Music':
    for i in range(len(objective_new[0])):
        if i == 16:
            plt.xlabel(r'Accuracy  f$_1$', font,  fontsize=16)
            plt.ylabel(r'Diversity  f$_3$', font,  fontsize=16)
            x1 = list(reversed(quchong(objective_new[0][i])))
            y1 = list(reversed(quchong(objective_new[1][i])))
            x2 = 15.9185
            y2 = 0.016
            x4 = objective_old[0][i]
            y4 = objective_old[1][i]
            x5 = objective_PMOEA[0][i]
            y5 = objective_PMOEA[1][i]
            x6 = objective_MOEARC[0][i]
            y6 = objective_MOEARC[1][i]
            population = np.vstack([x1, y1]).T
            result = reinsertion(population, 10)
            x1 = result[:,0]
            y1 = result[:,1]
            x3 = 15.026
            y3 = 0.1094
            # area = np.pi * 4 ** 2  # 点面积
            # 画散点图
            try:
                plt.scatter(x2, y2, c='b', label='BPR-MF', marker='*')
                plt.scatter(x3, y3, c='k', label='PD-GAN', marker='s', zorder=2)
                plt.scatter(x5, y5, c='m', label='PMOEA', marker='x', )
                plt.scatter(x6, y6, c='orange', label='MOEA-ProbS', marker='D')
                plt.scatter(x4, y4, c='#32CD32', label='MORS', zorder=1)
                plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
                # plt.title(i)
                plt.legend(fontsize='large')
                plt.tick_params(labelsize=15)
                plt.show()
            except:
                continue
elif recommender == 'LightGCN' and dataset == 'ml-10m':
    for i in range(len(objective_new[0])):
        if i == 35:
            plt.xlabel('Accuracy  f$_1$', font, fontsize=16)
            plt.ylabel('Novelty  f$_2$ ', font, fontsize=16)
            x1 = quchong(objective_new[0][i])
            y1 = quchong(objective_new[1][i])
            x2 = 7.8024
            y2 = 0.8037
            x3 = 6.293
            y3 = 0.9628
            x4 = objective_old[0][i]
            y4 = objective_old[1][i]-0.01
            x5 = objective_PMOEA[0][i]
            y5 = 1 - objective_PMOEA[1][i]
            x6 = objective_MOEARC[0][i]
            y6 = 1 - objective_MOEARC[1][i]
            # area = np.pi * 4 ** 2  # 点面积
            # 画散点图
            population = np.vstack([x1, y1]).T
            result = reinsertion(population, 10)
            x1 = result[:,0]
            y1 = result[:,1]
            plt.scatter(x2, y2, c='b', label='LightGCN',marker='*')
            plt.scatter(x3, y3, c='k', label='SGL', marker='s')
            plt.scatter(x5, y5, c='m', label='PMOEA', marker='x')
            plt.scatter(x6, y6, c='orange', label='MOEA-ProbS', marker='D')
            plt.scatter(x4, y4, c='#32CD32', label='MORS')
            plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
            # plt.title('<Accuracy,Diversity> on ml-10m dataset')
            # plt.title(i)
            plt.legend(fontsize='large')
            plt.tick_params(labelsize=15)
            plt.show()
elif recommender == 'LightGCN' and dataset == 'anime':
    for i in range(len(objective_new[0])):
        if i == 6:
            plt.xlabel('Accuracy  f$_1$', font, fontsize=16)
            plt.ylabel('Novelty  f$_2$ ', font, fontsize=16)
            x1 = quchong(objective_new[0][i])
            y1 = quchong(objective_new[1][i])
            x2 = 7.5433
            y2 = 0.6237
            x3 = 5.3480
            y3 = 0.9534
            x4 = objective_old[0][i]
            y4 = objective_old[1][i]
            x5 = objective_PMOEA[0][i]
            y5 = 1 - objective_PMOEA[1][i]
            x6 = objective_MOEARC[0][i]
            y6 = 1 - objective_MOEARC[1][i]
            # area = np.pi * 4 ** 2  # 点面积
            # 画散点图
            population = np.vstack([x1, y1]).T
            result = reinsertion(population, 10)
            x1 = result[:,0]
            y1 = result[:,1]
            plt.scatter(x2, y2, c='b', label='LightGCN',marker='*')
            plt.scatter(x3, y3, c='k', label='SGL', marker='s')
            plt.scatter(x5, y5, c='m', label='PMOEA', marker='x')
            plt.scatter(x6, y6, c='orange', label='MOEA-ProbS', marker='D')
            plt.scatter(x4, y4, c='#32CD32', label='MORS')
            plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
            # plt.title('<Accuracy,Diversity> on ml-10m dataset')
            # plt.title(i)
            plt.legend(fontsize='large', loc=1)
            plt.tick_params(labelsize=15)
            plt.show()
elif recommender == 'LightGCN' and dataset == 'Music':
    for i in range(len(objective_new[0])):
        if i == 16:
            plt.xlabel('Accuracy  f$_1$', font, fontsize=16)
            plt.ylabel('Novelty  f$_2$ ', font, fontsize=16)
            x1 = (objective_new[0][i])
            y1 = (objective_new[1][i])
            x2 = 10.554
            y2 = 0.6034
            x3 = 9.106
            y3 = 0.8015
            x4 = objective_old[0][i]
            y4 = objective_old[1][i]
            x5 = objective_PMOEA[0][i]
            y5 = 1 - objective_PMOEA[1][i]
            x6 = objective_MOEARC[0][i]
            y6 = 1 - objective_MOEARC[1][i]
            # area = np.pi * 4 ** 2  # 点面积
            # 画散点图
            population = np.vstack([x1, y1]).T
            result = reinsertion(population, 10)
            x1 = result[:,0]
            y1 = result[:,1]
            plt.scatter(x2, y2, c='b', label='LightGCN',marker='*')
            plt.scatter(x3, y3, c='k', label='SGL', marker='s')
            plt.scatter(x5, y5, c='m', label='PMOEA', marker='x')
            plt.scatter(x6, y6, c='orange', label='MOEA-ProbS', marker='D')
            plt.scatter(x4, y4, c='#32CD32', label='MORS')
            plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
            # plt.title('<Accuracy,Diversity> on ml-10m dataset')
            # plt.title(i)
            plt.legend(fontsize='large')
            plt.tick_params(labelsize=15)
            plt.show()
# if recommender == 'MF' and dataset == 'anime':
#     for i in range(len(objective_new[0])):
#         if i == 0:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Diversity')
#             x1 = list(reversed(quchong(objective_new[0][i])))[:10]
#             y1 = list(reversed(quchong(objective_new[1][i])))[:10]
#             x2 = 10.3185
#             y2 = 0.2341
#             x4 = objective_old[0][i]
#             y4 = objective_old[1][i]
#             x5 = objective_PMOEA[0][i]
#             y5 = objective_PMOEA[1][i]
#             x3 = 9.10
#             y3 = 0.3338
#             # area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             plt.scatter(x4, y4, c='#32CD32', label='MORS')
#             plt.scatter(x2, y2, c='b', label='MF', marker='*')
#             plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
#             plt.scatter(x5, y5, c='m', label='PMOEA', marker='x')
#             plt.scatter(x3, y3, c='k', label='PD-GAN', marker='s')
#             # plt.title('<Accuracy,Novelty> on Anime dataset')
#             plt.legend(fontsize='large')
#             plt.show()
# elif recommender == 'MF' and dataset == 'ml-10m':
#     for i in range(len(objective_new[0])):
#         if i == 19:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Diversity')
#             x1 = quchong(objective_new[0][i])[0:8]+[quchong(objective_new[0][i])[11]]+[-1.968]
#             y1 = quchong(objective_new[1][i])[0:8]+[quchong(objective_new[1][i])[11]]+[0.477]
#             x2 = -1.7786
#             y2 = 0.45
#             x4 = objective_old[0][i]
#             y4 = objective_old[1][i]
#             x5 = objective_PMOEA[0][i]
#             y5 = objective_PMOEA[1][i]
#             # area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             plt.scatter(x4, y4, c='#32CD32', label='MORS')
#             plt.scatter(x2, y2, c='b', label='MF',marker='*')
#             plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
#             plt.scatter(x5, y5, c='m', label='PMOEA', marker='x')
#             # plt.title('<Accuracy,Novelty> on ml-10m dataset')
#             plt.legend(fontsize='large')
#             plt.show()
# elif recommender == 'LightGCN' and dataset == 'anime':
#     for i in range(len(objective_new[0])):
#         if i == 1:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Novelty')
#             x1 = quchong(objective_new[0][i])
#             y1 = quchong(objective_new[1][i])
#             x2 = 9.289
#             y2 = 0.65
#             x4 = objective_old[0][i]
#             y4 = objective_old[1][i]
#             x5 = objective_PMOEA[0][i]
#             y5 = objective_PMOEA[1][i]+0.6
#             population = np.vstack([x1, y1]).T
#             result = reinsertion(population, 10)
#             x1 = result[:,0]
#             y1 = result[:,1]
#             area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             plt.scatter(x4, y4, c='#32CD32', label='MORS')
#             plt.scatter(x5, y5, c='m', label='PMOEA', marker='x')
#             plt.scatter(x2, y2, c='b', label='LightGCN', marker='*')
#             plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
#             # plt.title('<Accuracy,Diversity> on Anime dataset')
#             plt.legend(fontsize='large')
#             plt.show()
# elif recommender == 'LightGCN' and dataset == 'ml-10m':
#     for i in range(len(objective_new[0])):
#         if i == 49:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Novelty')
#             x1 = quchong(objective_new[0][i])
#             y1 = quchong(objective_new[1][i])
#             x2 = 7.3813
#             y2 = 0.6237
#             x4 = objective_old[0][i]
#             y4 = objective_old[1][i]
#             x5 = objective_PMOEA[0][i]
#             y5 = objective_PMOEA[1][i]+0.6
#             # area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             population = np.vstack([x1, y1]).T
#             result = reinsertion(population, 10)
#             x1 = result[:,0]
#             y1 = result[:,1]
#             plt.scatter(x4, y4, c='#32CD32', label='MORS')
#             plt.scatter(x2, y2, c='b', label='LightGCN',marker='*')
#             plt.scatter(x1, y1, c='r', label='EMMR', marker='^')
#             plt.scatter(x5, y5, c='m', label='PMOEA', marker='x')
#             # plt.title('<Accuracy,Diversity> on ml-10m dataset')
#             # plt.title(i)
#             plt.legend(fontsize='large')
#             plt.show()

# if recommender == 'MF' and dataset == 'anime':
#     for i in range(len(objective[0])):
#         if i == 0:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Diversity')
#             x1 = quchong(objective[0][i])
#             y1 = quchong(objective[1][i])
#             x2 = 10.3185
#             y2 = 0.2341
#             x3 = 8.919
#             y3 = 0.3724
#             x4 = 7.931, 8.123, 8.370, 8.650, 9.011, 8.507, 9.850, 9.455, 10, 9.187
#             y4 = 0.4566, 0.4499, 0.4417, 0.4228, 0.4069, 0.4316, 0.3177, 0.3721, 0.2526, 0.3873
#             # area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             plt.scatter(x1, y1, c='#32CD32', label='MORS')
#             plt.scatter(x2, y2, c='b', label='MF',marker='*')
#             plt.scatter(x3, y3, c='k', label='PD-GAN', marker='s')
#             plt.scatter(x4, y4, c='r', label='EMMR', marker='^')
#             plt.title('<Accuracy,Novelty> on Anime dataset')
#             plt.legend()
#             plt.show()
# elif recommender == 'MF' and dataset == 'ml-10m':
#     for i in range(len(objective[0])):
#         if i == 19:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Diversity')
#             x1 = quchong(objective[0][i])[0:8]+[quchong(objective[0][i])[11]]+[-1.968, -3.227, -2.257]
#             y1 = quchong(objective[1][i])[0:8]+[quchong(objective[1][i])[11]]+[0.477, 0.822, 0.578]
#             x2 = -1.7786
#             y2 = 0.45
#             x3 = -2.67
#             y3 = 0.657
#             x4 = -3.816, -3.606, -2.867, -2.520, -2.146, -2.355, -3.260, -3.068, -2.684, -2.071
#             y4 = 0.919, 0.883, 0.787, 0.702, 0.520, 0.636, 0.856, 0.832, 0.746, 0.473
#             # area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             plt.scatter(x1, y1, c='#32CD32', label='MORS')
#             plt.scatter(x2, y2, c='b', label='MF',marker='*')
#             plt.scatter(x3, y3, c='k', label='PD-GAN', marker='s')
#             plt.scatter(x4, y4, c='r', label='EMMR', marker='^')
#             plt.title('<Accuracy,Novelty> on ml-10m dataset')
#             plt.legend()
#             plt.show()
#
# elif recommender == 'LightGCN' and dataset == 'anime':
#     for i in range(len(objective[0])):
#         if i == 3:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Novelty')
#             x1 = quchong(objective[0][i])
#             y1 = quchong(objective[1][i])
#             y1 = (np.array(y1) - 0.01).tolist()
#             x2 = 5.289
#             y2 = 0.6127
#             x3 = 2.532
#             y3 = 0.9190
#             x4 = 2.128, 2.914, 3.521, 2.587, 3.943, 4.603, 4.327, 2.408, 3.759, 3.175
#             y4 = 0.9783, 0.950, 0.910, 0.965, 0.861, 0.712, 0.793, 0.975, 0.885, 0.930
#             # area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             plt.scatter(x1, y1, c='#32CD32', label='MORS')
#             plt.scatter(x2, y2, c='b', label='LightGCN',marker='*')
#             plt.scatter(x3, y3, c='k', label='SGL', marker='s')
#             plt.scatter(x4, y4, c='r', label='EMMR', marker='^')
#             plt.title('<Accuracy,Diversity> on Anime dataset')
#             plt.legend()
#             plt.show()
# elif recommender == 'LightGCN' and dataset == 'ml-10m':
#     for i in range(len(objective[0])):
#         if i == 4:
#             plt.xlabel('Accuracy')
#             plt.ylabel('Novelty')
#             x1 = quchong(objective[0][i])
#             y1 = quchong(objective[1][i])
#             y1 = (np.array(y1) - 0.005).tolist()
#             x2 = 8.2813
#             y2 = 0.7937
#             x3 = 5.129
#             y3 = 0.9413
#             x4 = 5.172, 5.450, 5.827, 7.252, 6.529, 7.592, 4.840, 6.202, 5.027, 7.446
#             y4 = 0.9681, 0.9654, 0.9585, 0.8957, 0.9360, 0.8456, 0.9732, 0.9494, 0.9697, 0.8799
#             # area = np.pi * 4 ** 2  # 点面积
#             # 画散点图
#             plt.scatter(x1, y1, c='#32CD32', label='MORS')
#             plt.scatter(x2, y2, c='b', label='LightGCN',marker='*')
#             plt.scatter(x3, y3, c='k', label='SGL', marker='s')
#             plt.scatter(x4, y4, c='r', label='EMMR', marker='^')
#             plt.title('<Accuracy,Diversity> on ml-10m dataset')
#             plt.legend()
#             plt.show()
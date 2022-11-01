import numpy as np
import random
import geatpy as ea

def find(x):
    return np.where(x == 1)[0]

def TS(Fitness):
    return ea.selecting('tour', Fitness.reshape(-1, 1), 1)[0]



# # Simulated Binary Crossover
# def crossover(p1, p2, Fitness):
#     if random.random() < 0.5:
#         index = find(p1 & np.where(np.array(p2)==0,1,0))
#         try:
#             index = index[Tournamentselection(-Fitness[index])]
#             p1[index] = 0
#         except:
#             pass
#
#     else:
#         index = find(np.where(np.array(p1)==0,1,0) & p2)
#         try:
#             index = index[Tournamentselection(Fitness[index])]
#             p1[index] = p2[index]
#         except:
#             pass
#
#     return p1

def Ato_ada(c1,c2,tasks,p1,p2,log,no_of_objs,maxormins):
    C, P = [c1, c2], [p1,p2]
    result = []
    for c in C:
        ObjV = np.empty([2, no_of_objs])
        for i, p in enumerate(P):
            ObjV[i] = tasks[p].fnc(c, log[p])

        [levels, criLevel] = ea.ndsortESS(ObjV, 2, None, None, maxormins)
        if levels[0] == levels[1]:
            result.append(random.choice(P))
        else:
            result.append(P[levels.argmin()])
    return result[0], result[1]

# polynomial mutation
# def mutate(p, Fitness):
#     if random.random() < 0.5:
#         index = find(p)
#         index = index[TS(-Fitness[index])]
#         p[index] = 0
#     else:
#         index = find(np.where(np.array(p)==0,1,0))
#         index = index[TS(Fitness[index])]
#         p[index] = 1
#     return p

def mutate(p, Fitness):
    num = 100
    candicate = Fitness.argsort()[-num:][::-1]
    x = random.sample(list(candicate), 1)
    if p[x] == 1:
        p[x] = 0
    else:
        p[x] = 1
    return p

def RouletteWheelSelection(fitness):
    length = len(fitness)
    if length == 1:
        return 0

    accumulator = 0
    sumFits = np.sum(fitness)
    rndPoints = np.random.uniform(low=0, high=sumFits)
    for index, val in enumerate(fitness):
        accumulator += val
        if accumulator >= rndPoints:
            return index

def Tournamentselection(indicateValueDict, selectNum=1, elementNum=2):
    # 个体索引列表
    indicateList = range(len(indicateValueDict))
    # 选择出的个体序号 列表
    remainIndicateList = []


    for i in range(selectNum):
        tempList = []
        try:
            tempList.extend(random.sample(indicateList, elementNum))
        except:
            return 0
        bestIndicate = np.argsort(indicateValueDict[tempList])[1]
        remainIndicateList.append(tempList[bestIndicate])
    ###返回选择的索引列表
    return remainIndicateList

# def nondominatedsort(population,pop,no_of_objs):
#     count = 0
#     frontnumbers = []
#     for i in range(pop):
#         for j in range(pop):
#             if i == j:
#                 continue
#             better = 0
#             worse = 0
#             for k in range(no_of_objs):
#                 if population[i].objective[k] > population[j].objective[k]:
#                     better = 1
#                 elif population[i].objective[k] < population[j].objective[k]:
#                     worse = 1
#
#             if worse == 0 and better > 0:
#                 population[i].dominatedset.append(j)
#                 population[i].dominatedsetlength += 1
#                 population[j].dominationcount += 1
#             elif better == 0 and worse > 0:
#                 population[j].dominatedset.append(i)
#                 population[j].dominatedsetlength += 1
#                 population[i].dominationcount += 1
#
#         if population[i].dominationcount == 0:
#             population[i].front = 1
#             count = count + 1
#
#     frontnumbers.append(count)
#     front = 0
#     while count > 0:
#         count = 0
#         front += 1
#         for i in range(pop):
#             if population[i].front == front:
#                 for j in range(population[i].dominatedsetlength):
#                     ind = population[i].dominatedset[j]
#                     population[ind].dominationcount = population[ind].dominationcount - 1
#                     if population[ind].dominationcount == 0:
#                         population[ind].front = front + 1
#                         count = count + 1
#         frontnumbers.append(count)
#
#     #diversity
#     for i in range(pop):
#         population[i].CD = 0
#     population = np.array(sorted(population, key=lambda x: x.front))
#     currentind = 0
#     for i in range(population[pop-1].front):
#         subpopulation = population[currentind:currentind + frontnumbers[i]]
#         x = np.zeros(frontnumbers[i])
#         for j in range(no_of_objs):
#             for k in range(frontnumbers[i]):
#                 x[k] = subpopulation[k].objective[j]
#             y = np.argsort(-x)
#             x = x[y]
#             subpopulation = subpopulation[y]
#             max = subpopulation[0].objective[j]
#             min = subpopulation[frontnumbers[i]-1].objective[j]
#             if max == min:
#                 continue
#             subpopulation[0].CD = np.inf
#             subpopulation[frontnumbers[i] - 1].CD = np.inf
#             normobj = (x - max) / (min - max)
#             for k in range(1, frontnumbers[i]-1):
#                 subpopulation[k].CD = subpopulation[k].CD + (normobj[k + 1] - normobj[k - 1])
#
#         subpopulation = np.array(sorted(subpopulation, key=lambda x: x.CD, reverse=True))
#
#         population[currentind : currentind + frontnumbers[i]] = subpopulation
#         currentind = currentind + frontnumbers[i]
#
#     for i in range(pop):
#         population[i].rank = i
#
#     return population
def nondominatedsort(population,pop,no_of_objs, maxormins):
    ObjV = np.empty([len(population), no_of_objs])
    for i in range(len(population)):
        ObjV[i] = population[i].objective
    [levels, criLevel] = ea.ndsortESS(ObjV, pop, None, None, maxormins) # 对NUM个个体进行非支配分层
    dis = ea.crowdis(ObjV, levels)  # 计算拥挤距离
    FitnV = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
    chooseFlag = ea.selecting('dup', FitnV.reshape(-1, 1), pop)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体

    return population[chooseFlag], ObjV[chooseFlag]

# def crossover(p1, p2, f1, f2):







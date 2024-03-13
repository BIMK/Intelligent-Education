import numpy as np

def F_distance(FunctionValue,FrontValue):
    N, M = FunctionValue.shape
    CrowdDistance = np.zeros((1, N))
    # CrowdDistance_temp = CrowdDistance
    #差集： list(set(a)-set(b)), list(set(a).difference(set(b)))
    temp=np.unique(FrontValue)
    Fronts = temp[temp != np.inf]


    for f in range(len(Fronts)):
        Front = np.where(FrontValue == Fronts[f] )[1]# array 返回 两个值 ，前面的一维的坐标，后面是二维的坐标
        Fmax = np.max(FunctionValue[Front, :], axis=0)
        Fmin = np.min(FunctionValue[Front, :], axis=0)
        for i in range(M):
            Rank = FunctionValue[Front, i].argsort()
            CrowdDistance[0, Front[Rank[0]]] = np.inf
            CrowdDistance[0, Front[Rank[-1]]] = np.inf
            for j in range(1,len(Front)-1,1):
                CrowdDistance[0, Front[Rank[j]]] =  CrowdDistance[0, Front[Rank[j]]] + \
                                                         (FunctionValue[Front[Rank[j + 1]], i] - FunctionValue[Front[Rank[j-1]],i])/(Fmax[i]-Fmin[i])
    return CrowdDistance





import numpy as np
from EMO_public import F_distance
from EMO_public import NDsort

def F_EnvironmentSelect(Population,FunctionValue,N):
    FrontValue, MaxFront = NDsort.NDSort(FunctionValue, N)


    CrowdDistance = F_distance.F_distance(FunctionValue, FrontValue)


    Next = np.zeros((1, N), dtype="int64")
    # np.sum() 求矩阵all 元素和，
    NoN = np.sum(FrontValue<MaxFront)#可以用 int(np.sum(FrontValue<MaxFront,axis=1))代替
    Next[0, :NoN] = np.where(FrontValue <MaxFront)[1] # 满足条件的 索引 是个列向量，但可以赋值 给行向量
    # 拥挤距离 进行选取
    Last = np.where(FrontValue==MaxFront)[1]
    Rank =np.argsort(-(CrowdDistance[0,Last]))
    Next[0, NoN:] = Last[Rank[:N-NoN]]
    # print(np.unique(Next[0,NoN:]))




    FrontValue_temp =np.array( [FrontValue[0,Next[0,:]]])
    CrowdDistance_temp = np.array( [CrowdDistance[0, Next[0,:]]])
    # Population_temp = Population[Next[0,:],:]
    FunctionValue_temp = FunctionValue[Next[0,:], :]

    select_index = Next[0,:]
    Population_temp = [Population[i] for i in Next[0,:]]

    return Population_temp, FunctionValue_temp, FrontValue_temp, CrowdDistance_temp,select_index





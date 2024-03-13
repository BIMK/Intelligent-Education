import numpy as np

def F_mating(Population,FrontValue,CrowdDistance):
    N = len(Population)
    MatingPool = []
    MatingPool_index = []
    Rank = np.random.permutation(N)
    Pointer=0
    for i in range(0,N,2):
        k = [0, 0]
        for j in range(2):
            if Pointer+1 >= N:
                Rank = np.random.permutation(N)
                Pointer = 0

            p = Rank[Pointer]
            q = Rank[Pointer+1]
            if FrontValue[0,p] < FrontValue[0,q]:
                k[j] = p
            elif FrontValue[0,p] > FrontValue[0,q]:
                k[j] = q
            elif CrowdDistance[0,p] > CrowdDistance[0,q]:
                k[j] = p
            else:
                k[j] = q

            Pointer += 2

        MatingPool_index.extend(k[0:2])
        MatingPool.append(Population[k[0]])
        MatingPool.append(Population[k[1]])


    return MatingPool,MatingPool_index



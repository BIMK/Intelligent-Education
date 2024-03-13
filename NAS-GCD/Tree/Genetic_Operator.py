
import numpy as np
from copy import deepcopy

def ExchangeSubTree(Parent):
    parent_1,parent_2 = Parent[0],Parent[1]

    NumNode_1,NumNode_2 = parent_1.numNodes,parent_2.numNodes

    change_index_1 = np.random.randint(2,NumNode_1+1) # node 1 is root node (do not participate in exchange), node NumNode_1 is the last node
    change_index_2 = np.random.randint(2,NumNode_2+1) # based on level travel

    subTree_1, subTree_2 = parent_1.get_subTree(change_index_1),parent_2.get_subTree(change_index_2)

    parent_1.set_subTree(subTree_1.No, subTree_2)
    parent_2.set_subTree(subTree_2.No, subTree_1)

    individual_1, individual_2 = deepcopy(parent_1),deepcopy(parent_2)

    individual_1.After_Genetic()
    individual_2.After_Genetic()

    return individual_1,individual_2

def Deletion(Parent): # Delete the non-root node (from 2 to NumNode)
    parent_1,parent_2 = Parent[0],Parent[1]
    NumNode_1,NumNode_2 = parent_1.numNodes,parent_2.numNodes

    delete_index_1 = np.random.randint(2,NumNode_1+1) # node 1 is root node (do not participate in Deletion), node NumNode_1 is the last node
    delete_index_2 = np.random.randint(2,NumNode_2+1) # based on level travel

    parent_1.Deletion(delete_index_1)
    parent_2.Deletion(delete_index_2)

    individual_1, individual_2 = deepcopy(parent_1),deepcopy(parent_2)
    return individual_1,individual_2


def Replacement(Parent): # Delete the non-root node (from 2 to NumNode)
    parent_1,parent_2 = Parent[0],Parent[1]
    NumNode_1,NumNode_2 = parent_1.numNodes,parent_2.numNodes

    replace_index_1 = np.random.randint(1,NumNode_1+1) # node 1 is root node ( participate in Replacement), node NumNode_1 is the last node
    replace_index_2 = np.random.randint(1,NumNode_2+1) # based on level travel

    parent_1.Replacement(replace_index_1)
    parent_2.Replacement(replace_index_2)

    individual_1, individual_2 = deepcopy(parent_1),deepcopy(parent_2)
    return individual_1,individual_2

def Insert(Parent): # Delete the non-root node (from 2 to NumNode)
    parent_1,parent_2 = Parent[0],Parent[1]
    NumNode_1,NumNode_2 = parent_1.numNodes,parent_2.numNodes

    insert_index_1 = np.random.randint(1,NumNode_1+1) # node 1 is root node (participate in Insert), node NumNode_1 is the last node
    insert_index_2 = np.random.randint(1,NumNode_2+1) # based on level travel

    parent_1.Insertion(insert_index_1)
    parent_2.Insertion(insert_index_2)

    individual_1, individual_2 = deepcopy(parent_1),deepcopy(parent_2)
    return individual_1,individual_2

def Mutation(Parent):
    parent_1,parent_2 = Parent[0],Parent[1]
    NumNode_1,NumNode_2 = parent_1.numNodes,parent_2.numNodes

    insert_index_1 = np.random.randint(1,NumNode_1+1) # node 1 is root node (participate in Insert), node NumNode_1 is the last node
    insert_index_2 = np.random.randint(1,NumNode_2+1) # based on level travel

    parent_1.Mutation(insert_index_1)
    parent_2.Mutation(insert_index_2)

    individual_1, individual_2 = deepcopy(parent_1),deepcopy(parent_2)
    return individual_1,individual_2

def ChangeLeaf():
    pass







def Generate(Population,gen):
    offspring = []
    for i in range(0,len(Population),2):
        randi = np.random.randint(0,5) # Crossover or Mutation
        if Population[i].numNodes<2 or Population[i+1].numNodes<2: # if numNodes smaller than 2, Deletion and ExchangeSubTree should be avoid
            randi = np.random.randint(2,5)
        # randi = 0 # Crossover or Mutation
        if randi==0: # make crossover
            individual_1,individual_2 = ExchangeSubTree(deepcopy(Population[i:i+2]) )
        elif randi==1:
            individual_1,individual_2 = Deletion(deepcopy(Population[i:i+2]))
        elif randi ==2:
            individual_1,individual_2 = Replacement(deepcopy(Population[i:i+2]))
        elif randi ==3:
            individual_1,individual_2 = Insert(deepcopy(Population[i:i+2]))
        else:
            individual_1,individual_2 = Mutation(deepcopy(Population[i:i+2]))



        individual_1.gen = gen
        individual_2.gen = gen
        # individual_1.id = i
        # individual_2.id = i+1
        offspring.append(individual_1)
        offspring.append(individual_2)

    return offspring


def Generate_crossover_mutation(Population,gen):
    Offspring = []
    for i in range(0,len(Population),2):

        if Population[i].numNodes<2 or Population[i+1].numNodes<2: # if numNodes smaller than 2, Deletion and ExchangeSubTree should be avoid

            individual_1,individual_2 = deepcopy(Population[i]),deepcopy(Population[i+1])
        else:
            individual_1,individual_2 = ExchangeSubTree(deepcopy(Population[i:i+2]) )




        individual_1.gen = gen
        individual_2.gen = gen
        # individual_1.id = i
        # individual_2.id = i+1
        Offspring.append(individual_1)
        Offspring.append(individual_2)



    offspring = []
    for i in range(0,len(Offspring),2):
        randi = np.random.randint(1,5) # Crossover or Mutation
        if Offspring[i].numNodes<2 or Offspring[i+1].numNodes<2: # if numNodes smaller than 2, Deletion and ExchangeSubTree should be avoid
            randi = np.random.randint(2,5)
        # randi = 0 # Crossover or Mutation
        if  randi==1:
            individual_1,individual_2 = Deletion(deepcopy(Offspring[i:i+2]))
        elif randi ==2:
            individual_1,individual_2 = Replacement(deepcopy(Offspring[i:i+2]))
        elif randi ==3:
            individual_1,individual_2 = Insert(deepcopy(Offspring[i:i+2]))
        else:
            individual_1,individual_2 = Mutation(deepcopy(Offspring[i:i+2]))

        individual_1.gen = gen
        individual_2.gen = gen
        # individual_1.id = i
        # individual_2.id = i+1
        offspring.append(individual_1)
        offspring.append(individual_2)


    return offspring


def Generate_from_existing(Population,gen):
    offspring = []
    for i in range(0,len(Population),2):
        randi = np.random.randint(0,5) # Crossover or Mutation
        if Population[i].numNodes<2 or Population[i+1].numNodes<2: # if numNodes smaller than 2, Deletion and ExchangeSubTree should be avoid
            randi = np.random.randint(2,5)
        # randi = 0 # Crossover or Mutation
        # individual_1,individual_2 = Mutation(deepcopy(Population[i:i+2]) )
        if randi==0: # make crossover
            individual_1,individual_2 = ExchangeSubTree(deepcopy(Population[i:i+2]) )
        elif randi==1:
            individual_1,individual_2 = Deletion(deepcopy(Population[i:i+2]))
        elif randi ==2:
            individual_1,individual_2 = Replacement(deepcopy(Population[i:i+2]))
        elif randi ==3:
            individual_1,individual_2 = Insert(deepcopy(Population[i:i+2]))
        else:
            individual_1,individual_2 = Mutation(deepcopy(Population[i:i+2]))



        individual_1.gen = gen
        individual_2.gen = gen
        # individual_1.id = i
        # individual_2.id = i+1
        offspring.append(individual_1)
        offspring.append(individual_2)

    return offspring
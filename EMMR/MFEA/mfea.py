from MFEA.individual import Individual
# from MFEA.operators import crossover, mutate, RouletteWheelSelection, nondominatedsort, Ato_ada

from deap import base, creator, tools
import numpy as np
import geatpy as ea
import random
from tqdm import trange



def mfea(tasks, user, Data,
         gen = 50,
         rmp = 1,
         reps = 1,
         pop = 10,
         t_gen = 1,
         Len = 10,
         plot = False,
         operator = 'normal'):

    '''
    :param tasks: List of Task type, can not be empty
    :param pop: Integer, population size
    :param gen: Integer, generation
    :param rmp: Float, between 0 and 1
    :param reps: Integer, Repetition times
    :param plot: Boolean, True or false
    '''

    # assert len(tasks) >= 1 and pop % 2 == 0
    # if (pop % 2 != 0): pop += 1
    no_of_tasks = len(tasks)
    # fitness = np.asarray([x.reshape(-1,1) for x in fitness])




    for rep in range(reps):
        # print('Repetition: '+str(rep)+' :')
        population = np.asarray([Individual(Len, tasks) for _ in range(pop * no_of_tasks)])
        draw = None


        for i in range(no_of_tasks):
            for individual in population[i*pop:(i+1)*pop]:
                individual.skill_factor = i
                individual.evaluate(True, Data)
        no_of_obj = len(tasks[0].M)
        if no_of_obj == 2:
            maxormins = np.array([-1] * no_of_obj)
        if no_of_obj == 3:
            maxormins = np.array([-1, -1, 1])
        # toolbox = base.Toolbox()

        for generation in range(gen):
            child_index = {}
            for i in range(no_of_tasks):
                child_index[i] = []
            child = np.asarray([Individual(Len, tasks) for _ in range(pop * no_of_tasks)])
            parent = np.asarray([Individual(Len, tasks) for _ in range(pop * no_of_tasks)])
            for i in range(no_of_tasks):
                ObjV = np.empty([pop, no_of_obj])
                for j in range(pop):
                    ObjV[j] = population[i*pop+j].objective
                [levels, criLevel] = ea.ndsortESS(ObjV, pop, None, None,
                                                  maxormins)
                FitnV = (1 / levels).reshape(-1, 1)
                tmp_p = population[i*pop:(i+1)*pop][ea.selecting('tour', FitnV, pop)]
                parent[i*pop:(i+1)*pop] = tmp_p
            # inorder = np.random.permutation(pop)
            count = 0

            if generation % t_gen == 0 and generation != 0:
                pop_list = list(range(pop * no_of_tasks))
                while len(pop_list) != 0:
                    tmp1 = random.choice(pop_list)
                    pop_list.remove(tmp1)
                    tmp2 = random.choice(pop_list)
                    pop_list.remove(tmp2)
                    p1 = parent[tmp1]
                    p2 = parent[tmp2]
                    c1 = child[tmp1]
                    c2 = child[tmp2]
                    # c1.skill_factor = p1.skill_factor
                    # c2.skill_factor = p2.skill_factor


                    if(p1.skill_factor == p2.skill_factor or np.random.uniform()<rmp):

                        if operator == 'normal':
                            child1, child2 = crossover(p1.rnvec, p2.rnvec, p1.tasks[p1.skill_factor].rating, p2.tasks[p2.skill_factor].rating)

                            # child1, child2 = [toolbox.clone(ind) for ind in (p1.rnvec, p2.rnvec)]
                            # tools.cxUniform(child1, child2, 0.5)
                            # if p1.skill_factor == p2.skill_factor:
                            #     c1.rnvec = child1
                            #     c2.rnvec = child2
                            #     c1.skill_factor = p1.skill_factor
                            #     c2.skill_factor = p2.skill_factor
                            # else:
                            #     c1.rnvec = child1
                            #     c2.rnvec = child2
                            #     c1.skill_factor, c2.skill_factor = Ato_ada(c1.rnvec, c2.rnvec, tasks, p1.skill_factor, p2.skill_factor, Log, no_of_obj,maxormins)
                            #
                            #
                            # c1.rnvec = mutate(c1.rnvec, fitness[c1.skill_factor])
                            # c2.rnvec = mutate(c2.rnvec, fitness[c2.skill_factor])
                            #
                            # child_index[c1.skill_factor].append(tmp1)
                            # child_index[c2.skill_factor].append(tmp2)


                        elif operator == 'sparse':
                            c1.rnvec = crossover(p1.rnvec, p2.rnvec, fitness[p1.skill_factor])
                            c2.rnvec = crossover(p2.rnvec, p1.rnvec, fitness[p2.skill_factor])

                        # c1.rnvec = mutate(c1.rnvec, fitness[c1.skill_factor])
                        # c2.rnvec = mutate(c2.rnvec, fitness[c2.skill_factor])

                    else:
                        c1.skill_factor = p1.skill_factor
                        c1.rnvec = mutate(p1.rnvec, fitness[c1.skill_factor])

                        c2.skill_factor = p2.skill_factor
                        c2.rnvec = mutate(p2.rnvec, fitness[c2.skill_factor])
            else:
                for i in range(no_of_tasks):
                    pop_list = list(range(pop))
                    while len(pop_list) != 0:
                        tmp1 = random.choice(pop_list)
                        pop_list.remove(tmp1)
                        tmp2 = random.choice(pop_list)
                        pop_list.remove(tmp2)
                        p1 = parent[i*pop:(i+1)*pop][tmp1]
                        p2 = parent[i*pop:(i+1)*pop][tmp2]
                        c1 = child[i*pop:(i+1)*pop][tmp1]
                        c2 = child[i*pop:(i+1)*pop][tmp2]

                        child1, child2 = [toolbox.clone(ind) for ind in (p1.rnvec, p2.rnvec)]
                        tools.cxUniform(child1, child2, 0.5)
                        c1.rnvec = child1
                        c2.rnvec = child2
                        c1.skill_factor = p1.skill_factor
                        c2.skill_factor = p2.skill_factor

                        c1.rnvec = mutate(c1.rnvec, fitness[c1.skill_factor])
                        c2.rnvec = mutate(c2.rnvec, fitness[c2.skill_factor])

                        child_index[c1.skill_factor].append(i*pop + tmp1)
                        child_index[c2.skill_factor].append(i*pop + tmp2)




            for individual in child:
                individual.evaluate(False, Log[individual.skill_factor], Mi, fitness[individual.skill_factor].copy())
            # population = np.hstack((population, child))


            for i in range(no_of_tasks):

                tmp_p = np.hstack((population[i*pop:(i+1)*pop], child[child_index[i]]))
                population[i*pop:(i+1)*pop], tmp_obj = nondominatedsort(tmp_p, pop, no_of_obj, maxormins)
                [levels, criLevel] = ea.ndsortESS(tmp_obj, pop, None, None, maxormins)
                NDSet_obj = tmp_obj[np.where(levels == 1)[0]]
                if generation == gen - 1:
                    if i == 0:
                        fin_obj = tmp_obj
                    else:
                        fin_obj = np.vstack([fin_obj, tmp_obj])
                log[user[i]].append(ea.indicator.HV(NDSet_obj))

                



            if plot == True:
                current_obj = np.empty([pop, no_of_obj])
                for i in range(len(population)):
                    for j in range(no_of_obj):
                        current_obj[i][j] = population[i].objective[j]

                draw = ea.moeaplot(current_obj, 'objective values', False, draw, generation, gridFlag=True)
    # for i in range(no_of_tasks):

    #     Metrics = np.array([log[['hv'][0]]]).T
    #     ea.trcplot(Metrics, labels=[['hv']], titles=[['hv']])

    if plot == True:
        ea.moeaplot(current_obj, 'Pareto Front', saveFlag=False, gridFlag=True)

    return population, log, fin_obj
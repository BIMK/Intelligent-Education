import numpy as np
import math
import geatpy as ea
import random

class Individual(object):

    def __init__(self, D_multitask, tasks):
        self.dim = D_multitask
        self.tasks = tasks
        self.no_of_tasks = len(tasks)
        # self.rnvec = np.random.randint(0, 2, D_multitask)
        self.skill_factor = None
        self.CD = None
        self.rnvec = None


    def evaluate(self, need_init, Data):
        if self.skill_factor == None:
            raise ValueError("skill factor not set")
        if need_init:
            task = self.tasks[self.skill_factor]
            candicate = task.rating.keys()
            self.rnvec = random.sample(list(candicate),  self.dim)
            self.objective = task.fnc(self.rnvec, Data.Mi, Data.Si)
        else:
            task = self.tasks[self.skill_factor]
            self.objective = task.fnc(self.rnvec, Data.Mi, Data.Si)


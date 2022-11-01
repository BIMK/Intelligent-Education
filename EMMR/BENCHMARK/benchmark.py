from BENCHMARK.basic import Ackley, Griewank, Rastrigin, Rosenbrock, \
                                Schwefel, Sphere, Weierstrass
from BENCHMARK.task import Task

import scipy.io as sio
import os

def mat2python(filename, flags):
    path = os.path.abspath(os.path.dirname(__file__))
    file = path + filename
    data = sio.loadmat(file)
    names = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    parameters = []
    for i, flag in enumerate(flags):
        if flag!= None:
            name = names[i]
            parameters.append(data[name])
        else:
            parameters.append(None)
    return parameters

def CI_HS(filename = '\Tasks\CI_H.mat'):
    #  Complete Intersection and High Similarity (CI+HS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Griewank(M=params[2], opt=params[0])
    Task2 = Rastrigin(M=params[3], opt=params[1])
    tasks = [Task(50, Task1.fnc, -100, 100),
             Task(50, Task2.fnc, -50, 50)]
    return tasks

def CI_MS(filename = '\Tasks\CI_M.mat'):
    # Complete Intersection and Medium Similarity (CI+MS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Ackley(M=params[2], opt=params[0])
    Task2 = Rastrigin(M=params[3], opt=params[1])
    tasks = [Task(50, Task1.fnc, -50, 50),
             Task(50, Task2.fnc, -50, 50)]
    return tasks

def CI_LS(filename = '\Tasks\CI_L.mat'):
    #  Complete Intersection and Low Similarity
    flags = ['GO_Task1', None, 'Rotation_Task1',None]
    params = mat2python(filename, flags)
    Task1 = Ackley(M=params[2], opt=params[0])
    Task2 = Schwefel()
    tasks = [Task(50, Task1.fnc, -50, 50),
             Task(50, Task2.fnc, -500, 500)]
    return tasks

def PI_HS(filename = '\Tasks\PI_H.mat'):
    #  Partial Intersection and High Similarity (PI+HS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Rastrigin(M=params[2], opt=params[0])
    Task2 = Sphere(opt= params[1])
    tasks = [Task(50, Task1.fnc, -50, 50),
             Task(50, Task2.fnc, -100, 100)]
    return tasks

def PI_MS(filename = '\Tasks\PI_M.mat'):
    # Partial Intersection and Medium Similarity (PI+MS)
    flags = ['GO_Task1',None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Task1 = Ackley(M=params[2], opt=params[0])
    Task2 = Rosenbrock()
    tasks = [Task(50, Task1.fnc, -50, 50),
             Task(50, Task2.fnc, -50, 50)]
    return tasks

def PI_LS(filename = '\Tasks\PI_L.mat'):
    # Partial Intersection and Low Similarity (PI+LS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Ackley(M=params[2], opt=params[0])
    Task2 = Weierstrass(M=params[3], opt=params[1])
    tasks = [Task(50, Task1.fnc, -50, 50),
             Task(25, Task2.fnc, -0.5, 0.5)]
    return tasks

def NI_HS(filename = r'\Tasks\NI_H.mat'):
    # No Intersection and High Similarity
    flags = [None, 'GO_Task2', None, 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Rosenbrock()
    Task2 = Rastrigin(M=params[3], opt=params[1])
    tasks = [Task(50, Task1.fnc, -50, 50),
             Task(50, Task2.fnc, -50, 50)]
    return tasks

def NI_MS(filename = r'\Tasks\NI_M.mat'):
    # No Intersection and Medium Similarity (NI+MS)
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Task1 = Griewank(M=params[2], opt=params[0])
    Task2 = Weierstrass(M=params[3], opt=params[1])
    tasks = [Task(50, Task1.fnc, -100, 100),
             Task(50, Task2.fnc, -0.5, 0.5)]
    return tasks

def NI_LS(filename = r'\Tasks\NI_L.mat'):
    # No Intersection and Low Similarity (NI+LS)
    flags = ['GO_Task1',None, 'Rotation_Task1',None]
    params = mat2python(filename, flags)
    Task1 = Rastrigin(M=params[2], opt=params[0])
    Task2 = Schwefel()
    tasks = [Task(50, Task1.fnc, -50, 50),
             Task(50, Task2.fnc, -500, 500)]
    return tasks
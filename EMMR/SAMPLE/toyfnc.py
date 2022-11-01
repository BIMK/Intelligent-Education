import numpy as np

def ackley(x=None, a = 20,b = 0.2,c = None):
    if c is None:
        c = 2.0*np.pi

    firstSum = 0.0
    secondSum = 0.0

    #Refactor it using mapping
    for value in x:
        firstSum += value**2.0
        secondSum += np.cos(c*value)

    firstExp = np.exp([-b * np.sqrt(firstSum/len(x))])[0]
    secondExp = np.exp([secondSum/len(x)])[0]

    return -a * firstExp - secondExp + a + np.exp(1)

def rastrigin(x):
    Sum=0
    for value in x:
        Sum += value ** 2 - 10 * np.cos(2 * np.pi * value) + 10
    return Sum

def sphere(x):
    Sum=0.0
    for value in x:
        Sum+=value**2
    return Sum
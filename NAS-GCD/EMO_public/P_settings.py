def P_settings(Algorithm, Problem, M):
    # [Generations, N] = set_problem(Problem, M)
    # Generations = [1000 700 250 250 250 250 250 250 250]
    # N = [100 105 120 126 132 112 156 90 275];
    Generations = 1000
    N = 100
    N = N*(M-1)
    return Generations, N



def set_problem(Problem, M):
    pass
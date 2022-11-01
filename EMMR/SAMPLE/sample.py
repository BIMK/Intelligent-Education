from MFEA.mfea import mfea
from BENCHMARK.task import Task
from BENCHMARK.benchmark import CI_HS, CI_MS, CI_LS, PI_HS, PI_MS, PI_LS, NI_HS, NI_MS, NI_LS
from SAMPLE.toyfnc import ackley, sphere, rastrigin

if __name__=="__main__":
    # print('test')
    # tasks = [Task(50, ackley, 50, -50),
    #          Task(20, sphere, 50, -50)]
    #
    # TotalEvaluations, bestobj, bestind = mfea(tasks)
    # tasks = CI_HS()
    tasks = CI_HS()
    TotalEvaluations, bestobj, bestind = mfea(tasks, reps=1)
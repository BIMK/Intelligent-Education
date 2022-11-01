import numpy as np
import scipy.optimize as op

def fnceval(task, rnvec, p_il, method):
    d = task.dim
    nvars = rnvec[:d]
    minrange = task.lb[:d]
    maxrange = task.ub[:d]
    y = maxrange - minrange
    vars = y * nvars + minrange
    if np.random.uniform() <= p_il:
        res = op.minimize(task.fnc, vars, method=method)
        x = res.x
        nvars = (x - minrange)/ y
        m_nvars = nvars
        m_nvars[m_nvars < 0] = 0
        m_nvars[m_nvars > 1] = 1
        if (m_nvars!=nvars).shape[0] != 0:
            nvars = m_nvars
            x = y * nvars + minrange
            objective = task.fnc(x)
        rnvec[: d]=nvars
        funcCount = res.nfev
    else:
        x = vars
        objective = task.fnc(x)
        funcCount = 1
    return rnvec, objective, funcCount
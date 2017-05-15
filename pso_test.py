from pyswarm import pso

def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1*(x2**2)

def con(x):
    x1 = x[0]
    x2 = x[1]
    return [x1 + x2 - 75]

lb = [0.0, 0.0]
ub = [100.0,100.0]

xopt, fopt = pso(banana, lb, ub, f_ieqcons=con, maxiter=1000)


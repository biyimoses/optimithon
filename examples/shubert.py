from Optimithon import Base
from Optimithon import QuasiNewton
from numpy import array, cos
from scipy.optimize import minimize

fun = lambda x: sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * \
    sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])
cons = (
    {'type': 'ineq', 'fun': lambda x: 100 - x[0]**2},
    {'type': 'ineq', 'fun': lambda x: 100 - x[1]**2})
x0 = array((1., -1.))
sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)
print("solution according to 'COBYLA':")
print(sol1)
print("solution according to 'SLSQP':")
print(sol2)

OPTIM = Base(fun, ineq=[lambda x: 100. - x[i]**2 for i in range(2)],
             br_func='Carrol',
             penalty=1.e6,
             method=QuasiNewton, x0=x0,  # max_lngth=100.,
             t_method='Cauchy',  # 'Cauchy_x', 'ZeroGradient',
             dd_method='Gradient',
             # 'Newton', 'SR1', 'HestenesStiefel', 'PolakRibiere', 'FletcherReeves', 'Gradient', 'DFP', 'BFGS', 'Broyden', 'DaiYuan'
             ls_method='Backtrack',  # 'BarzilaiBorwein', 'Backtrack',
             ls_bt_method='Armijo',  # 'Armijo', 'Goldstein', 'Wolfe', 'BinarySearch'
             )
OPTIM.Verbose = False
OPTIM.MaxIteration = 1500
OPTIM()
print("================================================="*2)
print(OPTIM.solution)

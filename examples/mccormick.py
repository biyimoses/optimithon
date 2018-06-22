from Optimithon import Base
from Optimithon import QuasiNewton
from numpy import array, sin, pi
from scipy.optimize import minimize

fun = lambda x: sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1.
x0 = array((0., 0.))
print(fun(x0))
sol1 = minimize(fun, x0, method='COBYLA')
sol2 = minimize(fun, x0, method='SLSQP')
print("solution according to 'COBYLA':")
print(sol1)
print("solution according to 'SLSQP':")
print(sol2)

OPTIM = Base(fun,
             method=QuasiNewton, x0=x0,  # max_lngth=100.,
             t_method='Cauchy_x',  # 'Cauchy_x', 'ZeroGradient',
             dd_method='BFGS',
             # 'Newton', 'SR1', 'HestenesStiefel', 'PolakRibiere', 'FletcherReeves', 'Gradient', 'DFP', 'BFGS', 'Broyden', 'DaiYuan'
             ls_method='Backtrack',  # 'BarzilaiBorwein', 'Backtrack',
             ls_bt_method='Armijo',  # 'Armijo', 'Goldstein', 'Wolfe', 'BinarySearch'
             )
OPTIM.Verbose = False
OPTIM.MaxIteration = 1500
OPTIM()
print("==========================" * 4)
print(OPTIM.solution)

from Optimithon import Base
from Optimithon import QuasiNewton
from numpy import array
from scipy.optimize import minimize

NumVars = 9
fun = lambda x: sum([100 * (x[i + 1] - x[i]**2)**2 +
                     (1 - x[i])**2 for i in range(NumVars - 1)])
cons = [
    {'type': 'ineq', 'fun': lambda x: 9 - x[i]**2} for i in range(NumVars)]
x0 = array([0 for _ in range(NumVars)])
sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)
print("solution according to 'COBYLA':")
print(sol1)
print("solution according to 'SLSQP':")
print(sol2)

OPTIM = Base(fun, ineq=[lambda x: 9 - x[i]**2 for i in range(NumVars)],
             br_func='Carrol',
             penalty=1.e6,
             method=QuasiNewton, x0=x0,  # max_lngth=100.,
             t_method='Cauchy',  # 'Cauchy_x', 'ZeroGradient',
             dd_method='BFGS',
             # 'Newton', 'SR1', 'HestenesStiefel', 'PolakRibiere', 'FletcherReeves', 'Gradient', 'DFP', 'BFGS', 'Broyden', 'DaiYuan'
             ls_method='Backtrack',  # 'BarzilaiBorwein', 'Backtrack',
             ls_bt_method='Armijo',  # 'Armijo', 'Goldstein', 'Wolfe', 'BinarySearch'
             )
OPTIM.Verbose = False
OPTIM.MaxIteration = 1500
OPTIM()
print("================================================="*2)
print(OPTIM.solution)

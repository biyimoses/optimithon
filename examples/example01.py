from numpy import array
import numdifftools as nd
from Optimithon.base import Base
from Optimithon.NumericDiff import Simple
from Optimithon.QuasiNewton import QuasiNewton
from scipy.optimize import minimize


def f(x):
    from math import sin, cos
    return x[0] ** 2 - 3 * x[1] ** 3 + cos(x[1]) ** 2 * x[1] ** 4 - 10 * sin(x[0])


def jac(x):
    from math import sin, cos
    return array([2 * x[0] - 10 * cos(x[0]),
                  -9 * x[1] ** 2 - 2 * sin(x[1]) * cos(x[1]) * x[1] ** 4 + 4 * cos(x[1]) ** 2 * x[1] ** 3])


D = Simple()
x0 = array((3.2, 3.5))

f1 = lambda x: x[1] - 3.
f2 = lambda x: 4. - x[1]
f3 = lambda x: x[0] - 2.
f4 = lambda x: 4. - x[0]
OPTIM = Base(f, ineq=[f1, f2, f3, f4],
             br_func='Expn',
             penalty=1.e6,
             method=QuasiNewton, x0=x0,  # max_lngth=100.,
             t_method='Cauchy',  # 'Cauchy_x', 'ZeroGradient',
             dd_method='BFGS',
             # 'Newton', 'SR1', 'HestenesStiefel', 'PolakRibiere', 'FletcherReeves', 'Gradient', 'DFP', 'BFGS', 'Broyden', 'DaiYuan'
             ls_method='Backtrack',  # 'BarzilaiBorwein', 'Backtrack',
             ls_bt_method='Armijo',  # 'Armijo', 'Goldstein', 'Wolfe', 'BinarySearch'
             difftool=nd,
             jac=jac
             )
OPTIM.Verbose = False
OPTIM.MaxIteration = 1500
OPTIM()
print(OPTIM.solution)

scipymethods = ['COBYLA', 'SLSQP']
cns = (
    {'type': 'ineq', 'fun': f1}, {'type': 'ineq', 'fun': f2}, {'type': 'ineq', 'fun': f3}, {'type': 'ineq', 'fun': f4})
for mtd in scipymethods:
    try:
        print("Method: %s" % mtd)
        print(minimize(f, x0, method=mtd, constraints=cns))
    except:
        pass

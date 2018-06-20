from numpy import array
import numdifftools as nd
from Optimithon import base, QuasiNewton


def g(x):
    from math import cos
    return sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])


x0 = array((3., 4.5))
OPTIM = base.Base(g, method=QuasiNewton.QuasiNewton, x0=x0,
                  t_method='Cauchy_x',  # 'ZeroGradient',
                  dd_method='SR1',  # 'Gradient', 'SR1'
                  ls_method='Backtrack',
                  ls_bt_method='Armijo',
                  difftool=nd,
                  )  # , jac=jac)

OPTIM.Verbose = False
OPTIM()
print(OPTIM.solution)

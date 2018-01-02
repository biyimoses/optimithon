from numpy import array
import numdifftools as nd
def g(x):
    from math import cos
    return sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])


x0 = array((3., 4.5))
OPTIM = Base(g, method=QuasiNewton, x0=x0,
             t_method='Cauchy',  # 'Cauchy', #'ZeroGradient',
             dd_method='Gradient',  # 'HestenesStiefel', #'PolakRibiere', #'FletcherReeves',#'Gradient'
             ls_method='Backtrack',
             ls_bt_method='Armijo',  # 'Armijo', 'Goldstein' 'Wolfe'
             difftool=nd,
             )  # , jac=jac)

OPTIM.Verbose = False
OPTIM()
print(OPTIM.solution)

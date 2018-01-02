
def g(x):
    from math import cos
    return sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])


OPTIM = Base(g, method=QuasiNewton, x0=x0,
             t_method='ZeroGradient',
             dd_method='PolakRibiere', #''HestenesStiefel', #'PolakRibiere', #'FletcherReeves',#'Gradient'
             ls_method='Backtrack',
             ls_bt_method='Goldstein', #Goldstein' #''Wolfe'
             )  # , jac=jac)  # , difftool=nd)
# print(OPTIM.MaxIteration)
OPTIM.Verbose = False
OPTIM()
print(OPTIM.solution)

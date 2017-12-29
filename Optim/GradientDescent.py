from __future__ import print_function
from numpy import array, dot, sqrt
from base import OptimTemplate, Base
from NumericDiff import Simple


class LineSearch(object):
    def __init__(self, QNref, **kwargs):
        self.Ref = QNref
        # self.method = kwargs.pop('method', 'BarzilaiBorwein')
        self.method = kwargs.pop('method', 'Backtrack')
        self.Ref.MetaData['Step Size'] = self.method
        self.Arguments = kwargs

    def BarzilaiBorwein(self):
        x2 = self.Ref.x[-1]
        gx2 = self.Ref.gradients[-1]
        x1 = self.Ref.x[-2] if len(self.Ref.x) > 1 else None
        gx1 = self.Ref.gradients[-2] if len(self.Ref.x) > 1 else None
        if x1 is None:
            fx = self.Ref.obj_vals[-1]
            lngth = 1.
            t_x = x2 - lngth * gx2
            ft_x = self.Ref.objective(t_x)
            while ft_x > fx:
                lngth /= 2.
                t_x = x2 - lngth * gx2
                ft_x = self.Ref.objective(t_x)
        else:
            dif_x = x2 - x1
            dif_g = gx2 - gx1
            # lngth = sqrt(dot(gx2, gx2)) * dot(dif_x, dif_g) / dot(dif_g, dif_g)
            lngth = dot(dif_x, dif_g) / dot(dif_g, dif_g)
        return lngth

    def Armijo(self, alpha, ft_x, tx):
        fx = self.Ref.obj_vals[-1]
        gr = self.Ref.gradients[-1]
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        c1 = self.Arguments.pop('c1', .0001)
        # print(ft_x, fx + alpha * t)
        return ft_x > fx + alpha * c1 * dot(p, gr)

    def Wolfe(self, alpha, ft_x, tx):
        fx = self.Ref.obj_vals[-1]
        gr = self.Ref.gradients[-1]
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        c1 = self.Arguments.pop('c1', 1e-3)
        c2 = self.Arguments.pop('c2', .9)
        armijo = ft_x > fx + alpha * c1 * dot(p, gr)
        wolfe = (dot(p, self.Ref.grd(tx)) < c2 * dot(p, gr))
        return armijo or wolfe

    def StrongWolfe(self, alpha, ft_x, tx):
        fx = self.Ref.obj_vals[-1]
        gr = self.Ref.gradients[-1]
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        c1 = self.Arguments.pop('c1', .1)
        c2 = self.Arguments.pop('c2', .9)
        armijo = ft_x > fx + alpha * c1 * dot(p, gr)
        swolfe = abs(dot(p, self.Ref.grd(tx)) < c2 * abs(dot(p, gr)))
        return armijo or swolfe

    def Goldstein(self, alpha, ft_x, tx):
        fx = self.Ref.obj_vals[-1]
        gr = self.Ref.gradients[-1]
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        c1 = self.Arguments.pop('c1', 1. / 3.)
        g1 = fx + (1 - c1) * alpha * dot(gr, p) > ft_x
        g2 = ft_x > fx + c1 * alpha * dot(gr, p)
        return g1 or g2

    def Backtrack(self):
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        tau = self.Arguments.pop('tau', 3. / 4.)
        max_lngth = self.Arguments.pop('max_lngth', 1)
        alpha = max_lngth
        x = self.Ref.x[-1]
        t_x = x + alpha * p
        ft_x = self.Ref.objective(t_x)
        while self.Armijo(alpha, ft_x, t_x):
            alpha *= tau
            t_x = x + alpha * p
            ft_x = self.Ref.objective(t_x)
        return alpha

    def __call__(self, *args, **kwargs):
        return self.__getattribute__(self.method)()


class DescentDirection(object):
    def __init__(self, QNRef, **kwargs):
        self.Ref = QNRef
        self.method = kwargs.pop('method', 'HestenesStiefel')
        self.Ref.MetaData['Descent Direction'] = self.method

    def Gradient(self):
        direction = -self.Ref.gradients[-1]
        # direction /= sqrt(dot(direction, direction))
        self.Ref.directions.append(direction)
        return direction

    def FletcherReeves(self):
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients)>1 else None
        if gr1 is None:
            direction = -gr2
        else:
            beta_fr = dot(gr2, gr2)/dot(gr1, gr1)
            direction = -gr2 + beta_fr * self.Ref.directions[-1]
        self.Ref.directions.append(direction)
        return direction

    def PolakRibiere(self):
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients)>1 else None
        if gr1 is None:
            direction = -gr2
        else:
            beta_pr = dot(gr2, gr2-gr1)/dot(gr1, gr1)
            direction = -gr2 + beta_pr * self.Ref.directions[-1]
        self.Ref.directions.append(direction)
        return direction

    def HestenesStiefel(self):
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients)>1 else None
        if gr1 is None:
            direction = -gr2
        else:
            beta_hs = dot(gr2, gr2-gr1)/dot(gr2 - gr1, self.Ref.directions[-1])
            direction = -gr2 + beta_hs * self.Ref.directions[-1]
        self.Ref.directions.append(direction)
        return direction

    def __call__(self, *args, **kwargs):
        return self.__getattribute__(self.method)()


class Termination(object):
    def __init__(self, QNRef, **kwargs):
        self.Ref = QNRef
        self.method = kwargs.pop('method', 'Cauchy')
        self.Ref.MetaData['Termination Criterion'] = self.method

    def Cauchy(self):
        progress = abs(self.Ref.obj_vals[-1] - self.Ref.obj_vals[-2])
        if progress <= self.Ref.ErrorTolerance:
            self.Ref.Success = True
            self.Ref.termination_message = "Progress in objective values less than error tolerance (Cauchy condition)" % progress
            return True
        return False

    def ZeroGradient(self):
        from numpy import absolute
        gr_mx = max(absolute(self.Ref.gradients[-1]))
        if gr_mx <= self.Ref.ErrorTolerance:
            self.Ref.Success = True
            self.Ref.termination_message = "Reached a point whose Gradient is almost zero"
            return True
        return False

    def __call__(self, *args, **kwargs):
        if self.Ref.STEP == 0:
            return False
        elif self.Ref.STEP > self.Ref.MaxIteration:
            self.Ref.termination_message = "Maximum number of iterations reached"
            return True
        else:
            return self.__getattribute__(self.method)()


class QuasiNewton(OptimTemplate):
    def __init__(self, obj, **kwargs):
        from types import FunctionType, LambdaType
        from collections import OrderedDict
        # check `obj` to be a function
        assert type(obj) in [FunctionType, LambdaType], "`obj` must be a function (the objective function)"
        super(QuasiNewton, self).__init__(obj, **kwargs)
        self.x = [self.x0]
        self.obj_vals = [self.objective(self.x0)]
        # If the gradient is given
        self.grd = kwargs.pop('jac', None)
        # Else
        if self.grd is None:
            # If a method to find gradient is given
            difftool = kwargs.pop('difftool', Simple())
            self.grd = difftool.Gradient(self.objective)
        self.gradients = []
        self.directions = []
        self.MetaData = OrderedDict([('Family', "Quasi-Newton method")])
        # TBM
        self.LineSearch = LineSearch(self, **kwargs)
        self.DescentDirection = DescentDirection(self, **kwargs)
        self.Termination = Termination(self, **kwargs)
        # self.step_sizes = []
        self.custom_step_size = kwargs.pop('step_size', None)

    def iterate(self):
        x = self.x[-1]
        self.gradients.append(self.grd(x))
        ddirection = self.DescentDirection()
        step_size = self.LineSearch()
        n_x = x + step_size * ddirection
        self.x.append(n_x)
        self.obj_vals.append(self.objective(n_x))
        self.STEP += 1

    def terminate(self):
        return self.Termination()


def f(x):
    from math import sin, cos
    return x[0] ** 2 - 3 * x[1] ** 3 + cos(x[1]) ** 2 * x[1] ** 4 - 10 * sin(x[0])


def jac(x):
    from math import sin, cos
    return array([2 * x[0] - 10 * cos(x[0]),
                  -9 * x[1] ** 2 - 2 * sin(x[1]) * cos(x[1]) * x[1] ** 4 + 4 * cos(x[1]) ** 2 * x[1] ** 3])


NumVars = 5


def rosen(x):
    return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(NumVars - 1)])
    # return (1 - x[0]) ** 2 + 105. * (x[1] - x[0] ** 2) ** 2


import numdifftools as nd

D = Simple()
x0 = array((-1.3, .51, 1.5, .7, 0.))
x0 = array((1.3, 1.8))
# print(D.Hessian(f)(x0))
# print(nd.Hessian(f)(x0))
print(f(array([1.28041337, 2.15681331e+05])))
# G = GDTpl(f, init=(1., 1.))
OPTIM = Base(f, method=QuasiNewton, x0=x0, difftool=nd)
# print(OPTIM.MaxIteration)
OPTIM.Verbose = False
OPTIM.MaxIteration = 500
OPTIM()
print(OPTIM.solution)
scipymethods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']
from scipy.optimize import minimize
for mtd in scipymethods:
    try:
        print("Method: %s"%mtd)
        print(minimize(f, x0, method=mtd))
    except:
        pass
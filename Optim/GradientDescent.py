from __future__ import print_function
from numpy import array, dot
from base import OptimTemplate, Base
from NumericDiff import Simple


class LineSearch(object):
    def __init__(self, QNref, **kwargs):
        self.Ref = QNref
        self.method = kwargs.pop('method', 'BarzilaiBorwein')
        self.Ref.MetaData['Step Size'] = self.method

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
            lngth = dot(dif_x, dif_g) / dot(dif_g, dif_g)
        return lngth

    def __call__(self, *args, **kwargs):
        return self.__getattribute__(self.method)()


class DescentDirection(object):
    def __init__(self, QNRef, **kwargs):
        self.Ref = QNRef
        self.method = kwargs.pop('method', 'Gradient')
        self.Ref.MetaData['Descent Direction'] = self.method

    def Gradient(self):
        direction = self.Ref.gradients[-1]
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
        self.LineSearch = LineSearch(self)
        self.DescentDirection = DescentDirection(self)
        self.Termination = Termination(self)
        # self.step_sizes = []
        self.custom_step_size = kwargs.pop('step_size', None)

    def iterate(self):
        x = self.x[-1]
        self.gradients.append(self.grd(x))
        ddirection = self.DescentDirection()
        step_size = self.LineSearch()
        n_x = x - step_size * ddirection
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
# x0 = array((1., 1.))
# print(D.Hessian(f)(x0))
# print(nd.Hessian(f)(x0))

# G = GDTpl(f, init=(1., 1.))
OPTIM = Base(rosen, method=QuasiNewton, x0=x0)  # , difftool=nd)
# print(OPTIM.MaxIteration)
OPTIM.Verbose = False
OPTIM.MaxIteration = 500
OPTIM()
print(OPTIM.solution)
from scipy.optimize import minimize

print(minimize(rosen, x0))

r"""
'QuasiNewton' Module
====================================================
This module contains implementations of variations of unconstrained optimization methods known as Quasi-Newton methods.
A Quasi-Newton method is an iterative algorithm that approximates a local minima of the objective function.
Starting with a given initial point `x0`, each iteration consists of three major subprocedures:

    + Finding a descent direction (via `DescentDirection` class)
    + Finding the length of descent (using `LineSearch` class)
    + Check the termination condition (`Termination` class).

The right strategy to attempt an optimization problem must be pre-determined, otherwise, it uses a default set up to
solve the problem.
"""

from __future__ import print_function
from numpy import array, dot, identity
from base import OptimTemplate, Base
from excpt import *


class LineSearch(object):
    r"""
    This class provides the step length at each iteration. The value of `ls_bt_method` at initiation determines whether
    to use `'BarzilaiBorwein'` method or a variation of `'Backtrack'` (default). It accepts a control parameter `tau`
    and `max_lngth` at initiation. `tau` must be a positive real less than 1.
    The variation of the backtrack is then determined by the value of `ls_bt_method` which can be selected among
    the following:

        + 'Armijo': indicates *Armijo* condition and the parameter `c1` can be modified at initiation as well.
        + 'Wolfe': indicates *Wolfe* condition and the parameters `c1` and `c2` can be modified at initiation.
        + 'StrongWolfe': indicates *StrongWolfe* condition and the parameters `c1` and `c2` can be modified at initiation.
        + 'Goldstein':  indicates *Goldstein* condition and the parameter `c1` can be modified at initiation.
    """

    def __init__(self, QNref, **kwargs):
        self.Ref = QNref
        self.method = kwargs.pop('ls_method', 'Backtrack')
        self.ls_bt_method = kwargs.pop('ls_bt_method', 'Armijo')
        self.Ref.MetaData['Step Size'] = self.method
        if self.method == 'Backtrack':
            self.Ref.MetaData['Backtrack Stop Criterion'] = self.ls_bt_method
        if self.method not in dir(self):
            raise Undeclared("The method `%s` is not implemented for `LineSearch`" % self.method)
        if self.ls_bt_method not in dir(self):
            raise Undeclared("The method `%s` is not implemented for `LineSearch`" % self.ls_bt_method)
        self.Arguments = kwargs

    def BarzilaiBorwein(self):
        """
        Implementation of *Barzilai-Borwein*.

        :return: step length
        """
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

    def Armijo(self, alpha, ft_x, tx):
        r"""
        Implementation of *Armijo*.

        :param alpha: current candidate for step length
        :param ft_x: value of the objective at the candidate point
        :param tx: the candidate point
        :return: `True` or `False`
        """
        fx = self.Ref.obj_vals[-1]
        gr = self.Ref.gradients[-1]
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        c1 = self.Arguments.pop('c1', .0001)
        # print(ft_x, fx + alpha * t)
        return ft_x > fx + alpha * c1 * dot(p, gr)

    def Wolfe(self, alpha, ft_x, tx):
        r"""
        Implementation of *Wolfe*.

        :param alpha: current candidate for step length
        :param ft_x: value of the objective at the candidate point
        :param tx: the candidate point
        :return: `True` or `False`
        """
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
        r"""
        Implementation of *Strong Wolfe*.

        :param alpha: current candidate for step length
        :param ft_x: value of the objective at the candidate point
        :param tx: the candidate point
        :return: `True` or `False`
        """
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
        r"""
        Implementation of *Goldstein*.

        :param alpha: current candidate for step length
        :param ft_x: value of the objective at the candidate point
        :param tx: the candidate point
        :return: `True` or `False`
        """
        fx = self.Ref.obj_vals[-1]
        gr = self.Ref.gradients[-1]
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        c1 = self.Arguments.pop('c1', 1. / 3.)
        g1 = fx + (1 - c1) * alpha * dot(gr, p) > ft_x
        g2 = ft_x > fx + c1 * alpha * dot(gr, p)
        return g1 or g2

    def BinarySearch(self, alpha, ft_x, tx):
        fx = self.Ref.obj_vals[-1]
        bs = ft_x < fx
        return bs

    def Backtrack(self):
        r"""
        A generic implementation of *Backtrack*.

        :return: step length
        """
        p = self.Ref.directions[-1]
        # TODO: make sure it is in the (0, 1) range
        tau = self.Arguments.pop('tau', 3. / 4.)
        max_lngth = self.Arguments.pop('max_lngth', 1)
        alpha = max_lngth
        x = self.Ref.x[-1]
        t_x = x + alpha * p
        ft_x = self.Ref.objective(t_x)
        while self.__getattribute__(self.ls_bt_method)(alpha, ft_x, t_x):
            alpha *= tau
            t_x = x + alpha * p
            ft_x = self.Ref.objective(t_x)
        return alpha

    def __call__(self, *args, **kwargs):
        return self.__getattribute__(self.method)()


class DescentDirection(object):
    r"""
    Implements various descent direction methods for Quasi-Newton methods. The descent method can be determined at
    initiation using `dd_method` parameter. The following values are acceptable:

        + 'Gradient': (default) The steepest descent direction.
        + 'FletcherReeves': Fletcher-Reeves method.
        + 'PolakRibiere': Polak-Ribiere method.
        + 'HestenesStiefel': Hestenes-Stiefel method.
        + 'DFP': Davidon-Fletcher-Powell formula.
        + 'BFGS': Broyden-Fletcher-Goldfarb-Shanno algorithm.
        + 'Broyden': Broyden's method.
        + 'SR1': Symmetric rank-one method.

    To calculate derivatives, the `QuasiNewton` class uses the object provided as the value of the `difftool` variable
    at initiation.
    """

    def __init__(self, QNRef, **kwargs):
        self.Ref = QNRef
        self.method = kwargs.pop('dd_method', 'Gradient')
        if self.method not in dir(self):
            raise Undeclared("The method `%s` is not implemented for `DescentDirection`" % self.method)
        self.Ref.MetaData['Descent Direction'] = self.method

    def Gradient(self):
        r"""
        :return: the gradient at current point
        """
        direction = -self.Ref.gradients[-1]
        self.Ref.directions.append(direction)
        return direction

    def FletcherReeves(self):
        r"""
        :return: the descent direction determined by *Fletcher-Reeves* method
        """
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients) > 1 else None
        if gr1 is None:
            direction = -gr2
        else:
            beta_fr = dot(gr2, gr2) / dot(gr1, gr1)
            direction = -gr2 + beta_fr * self.Ref.directions[-1]
        self.Ref.directions.append(direction)
        return direction

    def PolakRibiere(self):
        r"""
        :return: the descent direction determined by *Polak-Ribiere* method
        """
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients) > 1 else None
        if gr1 is None:
            direction = -gr2
        else:
            beta_pr = dot(gr2, gr2 - gr1) / dot(gr1, gr1)
            direction = -gr2 + beta_pr * self.Ref.directions[-1]
        self.Ref.directions.append(direction)
        return direction

    def HestenesStiefel(self):
        r"""
        :return: the descent direction determined by *Hestenes-Stiefel* method
        """
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients) > 1 else None
        if gr1 is None:
            direction = -gr2
        else:
            denum = dot(gr2 - gr1, self.Ref.directions[-1])
            if denum == 0.:
                raise Error(
                    """
                    Last descent direction is orthogonal to the difference of gradients of last two steps. 
                    This causes a division by zero in `Hestenes-Stiefel`.
                    One can avoid this by either slightly changing the initial point or choosing a different 
                    descent direction strategy.""")
            beta_hs = dot(gr2, gr2 - gr1) / denum
            direction = -gr2 + beta_hs * self.Ref.directions[-1]
        self.Ref.directions.append(direction)
        return direction

    def DFP(self):
        r"""
        :return: the descent direction determined by *Davidon-Fletcher-Powell* formula
        """
        x2 = self.Ref.x[-1]
        x1 = self.Ref.x[-2] if len(self.Ref.x) > 1 else None
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients) > 1 else None
        if x1 is None:
            from numpy.linalg import inv
            Hk = inv(self.Ref.hes(x2))
            self.Ref.InvHsnAprx.append(Hk)
            direction = - dot(Hk, gr2)
        else:
            sk = x2 - x1
            n = sk.shape[0]
            yk = gr2 - gr1
            rk = 1. / dot(yk, sk)
            Hk = self.Ref.InvHsnAprx[-1]
            Hk1 = Hk - dot(dot(Hk, dot(yk.reshape(n, 1), yk.reshape(1, n))), Hk) / dot(yk.reshape(1, n),
                                                                                       dot(Hk, yk.reshape(n, 1))) + dot(
                sk.reshape(n, 1), sk.reshape(1, n)) * rk
            self.Ref.InvHsnAprx.append(Hk1)
            direction = - dot(Hk1, gr2)
        self.Ref.directions.append(direction)
        return direction

    def BFGS(self):
        r"""
        :return: the descent direction determined by *Broyden-Fletcher-Goldfarb-Shanno* algorithm
        """
        x2 = self.Ref.x[-1]
        x1 = self.Ref.x[-2] if len(self.Ref.x) > 1 else None
        n = x2.shape[0]
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients) > 1 else None
        if x1 is None:
            from numpy.linalg import inv
            Hk = identity(n)
            # Hk = inv(self.Ref.hes(x2))
            self.Ref.InvHsnAprx.append(Hk)
            direction = - dot(Hk, gr2)
        else:
            sk = x2 - x1
            yk = gr2 - gr1
            rk = 1. / dot(yk, sk)
            Hk = self.Ref.InvHsnAprx[-1]
            I = identity(n)
            Hk1 = dot(dot(I - rk * dot(sk.reshape(n, 1), yk.reshape(1, n)), Hk),
                      I - rk * dot(yk.reshape(n, 1), sk.reshape(1, n))) + rk * dot(sk.reshape(n, 1), sk.reshape(1, n))
            self.Ref.InvHsnAprx.append(Hk1)
            direction = - dot(Hk1, gr2)
        self.Ref.directions.append(direction)
        return direction

    def Broyden(self):
        r"""
        :return: the descent direction determined by *Broyden's* method
        """
        x2 = self.Ref.x[-1]
        x1 = self.Ref.x[-2] if len(self.Ref.x) > 1 else None
        n = x2.shape[0]
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients) > 1 else None
        if x1 is None:
            from numpy.linalg import inv
            Hk = identity(n)
            # Hk = inv(self.Ref.hes(x2))
            self.Ref.InvHsnAprx.append(Hk)
            direction = - dot(Hk, gr2)
        else:
            sk = x2 - x1
            yk = gr2 - gr1
            Hk = self.Ref.InvHsnAprx[-1]
            Hk1 = Hk + dot((sk.reshape(n, 1) - dot(Hk, yk.reshape(n, 1))), dot(sk.reshape(1, n), Hk)) / dot(sk, dot(Hk,
                                                                                                                    yk.reshape(
                                                                                                                        n,
                                                                                                                        1)))
            self.Ref.InvHsnAprx.append(Hk1)
            direction = - dot(Hk1, gr2)
        self.Ref.directions.append(direction)
        return direction

    def SR1(self):
        r"""
        :return: the descent direction determined by *Symmetric rank-one* method
        """
        x2 = self.Ref.x[-1]
        x1 = self.Ref.x[-2] if len(self.Ref.x) > 1 else None
        n = x2.shape[0]
        gr2 = self.Ref.gradients[-1]
        gr1 = self.Ref.gradients[-2] if len(self.Ref.gradients) > 1 else None
        if x1 is None:
            from numpy.linalg import inv
            Hk = identity(n)
            #Hk = inv(self.Ref.hes(x2))
            self.Ref.InvHsnAprx.append(Hk)
            direction = - dot(Hk, gr2)
        else:
            sk = x2 - x1
            yk = gr2 - gr1
            Hk = self.Ref.InvHsnAprx[-1]
            J = (sk.reshape(n, 1) - dot(Hk, yk.reshape(n, 1)))
            Hk1 = Hk + dot(J, J.transpose()) / dot(yk.reshape(1, n), J)
            self.Ref.InvHsnAprx.append(Hk1)
            direction = - dot(Hk1, gr2)
        self.Ref.directions.append(direction)
        return direction

    def __call__(self, *args, **kwargs):
        return self.__getattribute__(self.method)()


class Termination(object):
    r"""
    Implements various termination criteria for Quasi-Newton loop. A particular termination method can be selected
    at initiation of the `Base` object by setting `t_method` with the name of the method as an string. The following
    termination criteria are implemented:

        + 'Cauchy': Checks of the changes in the values of the objective function are significant enough or not.
        + 'ZeroGradient': Checks if the gradient of the objective is close to zero or not.

    The value of the tolerated error is a property of `OptimTemplate` and hence can be modified as desired.
    """

    def __init__(self, QNRef, **kwargs):
        self.Ref = QNRef
        self.method = kwargs.pop('t_method', 'Cauchy')
        if self.method not in dir(self):
            raise Undeclared("The method `%s` is not implemented for `Termination`" % self.method)
        self.Ref.MetaData['Termination Criterion'] = self.method

    def Cauchy(self):
        r"""
        Checks if the values of the objective function form a Cauchy sequence or not.

        :return: `True` or `False`
        """
        progress = abs(self.Ref.obj_vals[-1] - self.Ref.obj_vals[-2])
        if progress <= self.Ref.ErrorTolerance:
            self.Ref.Success = True
            self.Ref.termination_message = "Progress in objective values less than error tolerance (Cauchy condition)"
            return True
        return False

    def Cauchy_x(self):
        r"""
        Checks if the sequence of points form a Cauchy sequence or not.

        :return: `True` or `False`
        """
        progress = max(abs(self.Ref.x[-1] - self.Ref.x[-2]))
        if progress <= self.Ref.ErrorTolerance:
            self.Ref.Success = True
            self.Ref.termination_message = "The progress in values of points is less than error tolerance (%f)" % progress
            return True
        return False

    def ZeroGradient(self):
        r"""
        Checks if the gradient vector is small enough or not.

        :return:  `True` or `False`
        """
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
    r"""
    This class hosts a family of first and second order iterative methods to solve an unconstrained optimization
    problem. The general schema follows the following steps:

        + Given the point :math:`x`, find a suitable descent direction :math:`p`.
        + Find a suitable length :math:`\alpha` for the direction :math:`p` such that :math:`x+\alpha p` results in an appropriate decrease in values of the objective.
        + Update :math:`x` to :math:`x+\alpha p` and repeat the above steps until a termination condition is satisfied.

    The initial value for `x` can be set at initiation by passing `x0=init_point` to the `Base` instance.
    There are various methods to determine the descent direction `p` at each step. The `DescentDirection` class
    implements a variety of these methods. To choose one of these methods one should pass the method by its known name
    at initiation simply by setting `dd_method='method name'`. This parameter will be passed to `DescentDirection`
    class (see the documentation for `DescentDirection`). Also, to determine a suitable value for :math:`alpha` various
    options are available the class `LineSearch` is responsible for handling the computation for :math:`alpha`.
    The parameters `ls_method` and `ls_bt_method` can be set at initiation to determine the details for line search.
    The termination condition also can vary and the desired condition can be determined by setting `t_method` at
    initiation which will be passed to the `Termination` class.
    Each of these classes may accept other parameters that can be set at initiation. To find out about those parameter
    see the corresponding documentation.
    """

    def __init__(self, obj, **kwargs):
        from types import FunctionType, LambdaType
        from collections import OrderedDict
        # check `obj` to be a function
        assert type(obj) in [FunctionType, LambdaType], "`obj` must be a function (the objective function)"
        super(QuasiNewton, self).__init__(obj, **kwargs)
        self.MetaData = OrderedDict([('Family', "Quasi-Newton method")])
        # TBM
        self.LineSearch = LineSearch(self, **kwargs)
        self.DescentDirection = DescentDirection(self, **kwargs)
        self.Termination = Termination(self, **kwargs)
        self.custom_step_size = kwargs.pop('step_size', None)

    def iterate(self):
        """
        This method updates the `iterate` method of the `OptimTemplate` by customizing the descent direction method
        as well as finding the descent step length. These method can be determined by the user.

        :return: None
        """
        x = self.x[-1]
        self.gradients.append(self.grd(x))
        ddirection = self.DescentDirection()
        step_size = self.LineSearch()
        # print(step_size, ddirection)
        n_x = x + step_size * ddirection
        self.x.append(n_x)
        self.obj_vals.append(self.objective(n_x))

    def terminate(self):
        """
        This method updates the `terminate` method of the `OptimTemplate` which is given by user.

        :return:  `True` or `False`
        """
        return self.Termination()


#"""
def f(x):
    from math import sin, cos
    return x[0] ** 2 - 3 * x[1] ** 3 + cos(x[1]) ** 2 * x[1] ** 4 - 10 * sin(x[0])


def jac(x):
    from math import sin, cos
    return array([2 * x[0] - 10 * cos(x[0]),
                  -9 * x[1] ** 2 - 2 * sin(x[1]) * cos(x[1]) * x[1] ** 4 + 4 * cos(x[1]) ** 2 * x[1] ** 3])


def g(x):
    from math import cos
    return sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])


NumVars = 5


def rosen(x):
    return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(NumVars - 1)])
    # return (1 - x[0]) ** 2 + 105. * (x[1] - x[0] ** 2) ** 2


import numdifftools as nd
from NumericDiff import Simple

D = Simple()
x0 = array((-1.3, .51, 1.5, .7, 0.))
x0 = array((5.5, 7.1))
x0 = array((1., 1.))
#x0 = array((1.78204552, 5.0806408))
x0 = array((-116.67964503, 1204.84914245))
x0 = array((-3.46611487, 1302.19015757))
print(f(x0))

OPTIM = Base(f, method=QuasiNewton, x0=x0,  # max_lngth=100.,
             t_method='Cauchy',  # 'Cauchy', 'ZeroGradient',
             dd_method='SR1',  # 'HestenesStiefel', 'PolakRibiere', 'FletcherReeves', 'Gradient', 'DFP', 'BFGS', 'Broyden'
             ls_method='Backtrack',  # 'BarzilaiBorwein', 'Backtrack',
             ls_bt_method='Wolfe',  # 'Armijo', 'Goldstein', 'Wolfe'
             # difftool=nd,
             )  # , jac=jac)
# print(OPTIM.MaxIteration)
OPTIM.Verbose = False
OPTIM.MaxIteration = 1500
OPTIM()
print(OPTIM.solution)
scipymethods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg',
                'trust-ncg']
from scipy.optimize import minimize

for mtd in scipymethods:
    try:
        print("Method: %s" % mtd)
        print(minimize(f, x0, method=mtd))
    except:
        pass
#"""

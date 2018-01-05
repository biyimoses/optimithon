r"""
'NumericDiff' Module
========================
This module provides very basic way to numerically approximate partial derivatives, gradient and Hessian of a function.
"""

from __future__ import print_function
from numpy import array

Infinitesimal = 1e-7


class Simple(object):
    r"""
    A simple class to calculate partial derivatives of a given function.
    """
    def __init__(self, **kwargs):
        pass

    def Diff(self, f, i=0):
        r"""
        :param f: a real valued function
        :param i: the index variable for differentiation
        :return: partial derivative of :math:`f` with respect to :math:'i^{th}` variable as a function.
        """
        assert i >= 0, "The variable index must be a positive integer for differentiation"

        def df(x):
            if i < len(x):
                x_ = x
                dx_p = list(x)
                dx_m = list(x)
                dx_p[i] = x[i] + Infinitesimal
                dx_m[i] = x[i] - Infinitesimal
                return (f(array(dx_p)) - f(array(dx_m))) / (2. * Infinitesimal)
            else:
                return 0.

        return df

    def Gradient(self, f):
        r"""
        :param f: a real valued function
        :return: a vector function that returns the gradient vector of `f` at each point.
        """
        def gf(x):
            n = len(x)
            gr = []
            for i in range(n):
                dx_p = list(x)
                dx_m = list(x)
                dx_p[i] = dx_p[i] + Infinitesimal
                dx_m[i] = dx_m[i] - Infinitesimal
                gr.append((f(dx_p) - f(dx_m)) / (2. * Infinitesimal))
            return array(gr)

        return gf

    def Hessian(self, f):
        r"""
        :param f: a real valued function
        :return: the Hessian matrix of :math:`f` at each point
        """
        def hsn(x):
            n = len(x)
            hs_mat = []
            for i in range(n):
                hs_row = []
                for j in range(n):
                    dx_pp = list(x)
                    dx_mm = list(x)
                    dx_pm = list(x)
                    dx_mp = list(x)
                    dx_pp[i] = dx_pp[i] + Infinitesimal
                    dx_pp[j] = dx_pp[j] + Infinitesimal
                    dx_pm[i] = dx_pm[i] + Infinitesimal
                    dx_pm[j] = dx_pm[j] - Infinitesimal
                    dx_mm[i] = dx_mm[i] - Infinitesimal
                    dx_mm[j] = dx_mm[j] - Infinitesimal
                    dx_mp[i] = dx_mp[i] - Infinitesimal
                    dx_mp[j] = dx_mp[j] + Infinitesimal
                    hs_row.append((f(dx_pp) - f(dx_mp) - f(dx_pm) + f(dx_mm)) / (4 * Infinitesimal ** 2))
                hs_mat.append(array(hs_row))
            return array(hs_mat)

        return hsn


D = Simple()
def f(x):
    return x[0]**2 - x[1]**3 +x[0]*x[1]**2
h = D.Hessian(f)
df0 = D.Diff(f, 0)
df1 = D.Diff(f, 1)
gf = D.Gradient(f)
x1 = array((1 , 1.))
x2 = array((2., 0.))
print(df0(x1), df1(x1), gf(x1))
from numpy import dot
x1 = array((1 , 1.))
print(x1, dot(h(x1), x1))
print(x1.transpose() - dot(x1.transpose(), h(x1)))
r"""
'NumericDiff' Module
========================
This module provides very basic way to numerically approximate partial derivatives, gradient and Hessian of a function.
"""

from __future__ import print_function

Infinitesimal = 1e-7


class Simple(object):
    r"""
    A simple class to calculate partial derivatives of a given function.
    Passing a value for `Infinitesimal` forces the calculations to be done according to the infinitesimal value
    provided by user. Otherwise, the default value (1.e-7) is used.
    """

    def __init__(self, **kwargs):
        self.Infinitesimal = kwargs.get('Infinitesimal', Infinitesimal)

    def Diff(self, f, i=0):
        r"""
        :param f: a real valued function
        :param i: the index variable for differentiation
        :return: partial derivative of :math:`f` with respect to :math:'i^{th}` variable as a function.
        """
        assert i >= 0, "The variable index must be a positive integer for differentiation"
        from numpy import array

        def df(x):
            try:
                n = len(x)
            except:
                n = 1
            if i < n:
                dx_p = list(x)
                dx_m = list(x)
                dx_p[i] = x[i] + self.Infinitesimal
                dx_m[i] = x[i] - self.Infinitesimal
                return (f(array(dx_p)) - f(array(dx_m))) / (2. * self.Infinitesimal)
            else:
                return 0.

        return df

    def Gradient(self, f):
        r"""
        :param f: a real valued function
        :return: a vector function that returns the gradient vector of `f` at each point.
        """
        from numpy import array

        def gf(x):
            try:
                n = len(x)
            except:
                n = 1
            gr = []
            for i in range(n):
                dx_p = list(x)
                dx_m = list(x)
                dx_p[i] = dx_p[i] + self.Infinitesimal
                dx_m[i] = dx_m[i] - self.Infinitesimal
                gr.append((f(dx_p) - f(dx_m)) / (2. * self.Infinitesimal))
            return array(gr)

        return gf

    def Hessian(self, f):
        r"""
        :param f: a real valued function
        :return: the Hessian matrix of :math:`f` at each point
        """
        from numpy import array
        def hsn(x):
            try:
                n = len(x)
            except:
                n = 1
            hs_mat = []
            for i in range(n):
                hs_row = []
                for j in range(n):
                    dx_pp = list(x)
                    dx_mm = list(x)
                    dx_pm = list(x)
                    dx_mp = list(x)
                    dx_pp[i] = dx_pp[i] + self.Infinitesimal
                    dx_pp[j] = dx_pp[j] + self.Infinitesimal
                    dx_pm[i] = dx_pm[i] + self.Infinitesimal
                    dx_pm[j] = dx_pm[j] - self.Infinitesimal
                    dx_mm[i] = dx_mm[i] - self.Infinitesimal
                    dx_mm[j] = dx_mm[j] - self.Infinitesimal
                    dx_mp[i] = dx_mp[i] - self.Infinitesimal
                    dx_mp[j] = dx_mp[j] + self.Infinitesimal
                    hs_row.append((f(dx_pp) - f(dx_mp) - f(dx_pm) + f(dx_mm)) / (4 * self.Infinitesimal ** 2))
                hs_mat.append(array(hs_row))
            return array(hs_mat)

        return hsn

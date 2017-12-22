from __future__ import print_function
from numpy import array
Infinitesimal = 1e-7


class Simple(object):
    def __init__(self, **kwargs):
        pass

    def diff(self, f, i=0):
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

    def gradient(self, f):

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
"""
D = Simple()
def f(x):
    return x[0]**2 - x[1]**3
df0 = D.diff(f, 0)
df1 = D.diff(f, 1)
gf = D.gradient(f)
x1 = array((1 , 1.))
x2 = array((2., 0.))
print(df0(x1), df1(x1), gf(x1))
"""
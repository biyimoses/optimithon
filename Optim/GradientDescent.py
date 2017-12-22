from __future__ import print_function
from numpy import array, dot
from base import OptimTemplate, Base
from NumericDiff import Simple


class GDTpl(OptimTemplate):
    def __init__(self, obj, **kwargs):
        # check `obj` to be a function
        super(GDTpl, self).__init__(obj, **kwargs)
        self.x = [self.x0]
        self.obj_vals = [self.objective(self.x0)]
        diffs = Simple()
        self.grd = diffs.gradient(self.objective)
        self.gradients = []
        # self.step_sizes = []
        self.custom_step_size = kwargs.pop('step_size', None)

    def step(self, x2, x1, gx2, gx1):
        if x1 is None:
            fx = self.obj_vals[-1]
            lngth = 1.
            t_x = x2 - lngth * gx2
            ft_x = self.objective(t_x)
            while ft_x > fx:
                lngth /= 2.
                t_x = x2 - lngth * gx2
                ft_x = self.objective(t_x)
        else:
            dif_x = x2 - x1
            dif_g = gx2 - gx1
            top = dot(dif_x, dif_g)
            den = dot(dif_g, dif_g)
            print(top, den)
            lngth = dot(dif_x, dif_g) / dot(dif_g, dif_g)
        return lngth

    def iterate(self):
        x = self.x[-1]
        self.gradients.append(self.grd(x))
        g_x = self.gradients[-1]
        x_p = None
        g_x_p = None
        if self.STEP > 0:
            x_p = self.x[-2]
            g_x_p = self.gradients[-2]
        gamma = self.step(x, x_p, g_x, g_x_p)
        print("____________")
        print(g_x)
        print(gamma)
        n_x = x - gamma * g_x
        print(n_x)
        print("++++++")
        self.x.append(n_x)
        self.obj_vals.append(self.objective(n_x))
        self.STEP += 1

    def terminate(self):
        if self.STEP == 0:
            return False
        progress = abs(self.obj_vals[-1] - self.obj_vals[-2])
        if progress <= self.ErrorTolerance:
            self.Success = True
            self.termination_message = "Progress less than error tolerance ()"
            return True
        elif self.STEP > self.MaxIteration:
            self.termination_message = "Maximum number of iterations reached"
            return True
        else:
            return False


def f(x):
    return x[0] ** 2 - 3*x[1] ** 3 + x[1] ** 4

D = Simple()
df = D.gradient(f)
x0 = array((1., 1.))
print(f(x0), df(x0))
#G = GDTpl(f, init=(1., 1.))
OPTIM = Base(f, method=GDTpl, x0=x0)
OPTIM()
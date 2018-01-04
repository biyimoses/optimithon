r"""
'base' Module
========================
"""

from __future__ import print_function
from NumericDiff import Simple


class OptimTemplate(object):
    def __init__(self, obj, **kwargs):
        from numpy import array
        self.MaxIteration = 100
        self.ErrorTolerance = 1e-7
        self.STEP = 0
        self.Success = False
        self.Terminate = False
        self.objective = None
        self.constraints = None
        self.iteration_message = None
        self.termination_message = None
        self.objective = obj
        self.solution = kwargs['solution']
        if 'init' in kwargs:
            self.x0 = array(kwargs['init'])
        elif 'x0' in kwargs:
            self.x0 = array(kwargs['x0'])
        else:
            self.x0 = None
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

    def iterate(self, **kwargs):
        pass

    def terminate(self, **kwargs):
        if self.STEP >= self.MaxIteration:
            self.Terminate = True
            self.termination_message = "Maximum iteration reached."
        self.Terminate = True
        return self.Terminate


class Base(object):
    r"""
    This is the base class that serves all the iterative optimization methods.
    An object derived from `Base` requires the following parameters:

    :param obj: *MANDATORY*- is a real valued function to be minimized.

    :param x0: an initial guess of the optimal point.

    :param method: the optimization class which implements `iterate` and `terminate` procedures (default: `OptimTemplate` that returns the value of the function at the initial point `x0`).

    :param Verbose: *Boolean*- If `True` prompts messages at every stage of the iteration as well as termination.

    The object then passes all other given parameters to the `method` class for further processes.
    When a termination condition is satisfied, the object fills the results in the `solution` attribute which is an
    instance of `Solution` class. The given class `method` can pass arbitrary pieces of information to the solution
    by modifying its `MetaData` dictionary.

    When an object `optim` of type `Base` initiated, the optimization process can be invoked by calling the object
    itself like a function::

        optim = Base(f, method=QuasiNewton, x0=init_point)
        optim()
        print(optim.solution)
    """

    def __init__(self, obj, **kwargs):
        self.x0 = None
        self.objective = obj
        self.constraints = []
        self.Verbose = True
        self.solution = Solution()
        _optimizer = kwargs.pop('method', OptimTemplate)
        self.optimizer = _optimizer(obj, solution=self.solution, **kwargs)

    def __call__(self, *args, **kwargs):
        from time import time
        start = time()
        # Iterate:
        while not self.optimizer.terminate(**kwargs):
            self.optimizer.iterate(**kwargs)
            self.optimizer.STEP += 1
            # Prompt the iteration message:
            if self.Verbose:
                if self.optimizer.iteration_message is not None:
                    print(self.optimizer.iteration_message)
        elapsed = (time() - start)
        # Prompt termination message:
        if self.Verbose:
            if self.optimizer.termination_message is not None:
                print(self.optimizer.termination_message)
        self.solution = self.optimizer.solution
        self.solution.NumIteration = self.optimizer.STEP
        self.solution.x = self.optimizer.x[-1]
        self.solution.objective = self.optimizer.obj_vals[-1]
        self.solution.success = self.optimizer.Success
        self.solution.message = self.optimizer.termination_message
        self.solution.RunTime = elapsed
        for itm in self.optimizer.MetaData:
            self.solution.__setattr__(itm, self.optimizer.MetaData[itm])

    def __setattr__(self, key, value):
        if key == 'MaxIteration':
            self.__dict__['optimizer'].MaxIteration = value
        else:
            self.__dict__[key] = value


class Solution(object):
    def __init__(self):
        self.__dict__['objective'] = None
        self.__dict__['NumIteration'] = 0
        self.__dict__['x'] = None
        self.__dict__['success'] = False
        self.__dict__['message'] = ""
        self.__dict__['RunTime'] = 0
        self.__dict__['attributes'] = ['objective', 'x', 'NumIteration', 'success', 'message', 'RunTime']

    def __setattr__(self, key, value):
        if key not in self.__dict__['attributes']:
            self.__dict__['attributes'].append(key)
        self.__dict__[key] = value

    def __repr__(self):
        output_line_tpl = """\t{}: {}\n"""
        output = ""
        for key in self.__dict__['attributes']:
            output += output_line_tpl.format(key, self.__dict__[key])
        return output

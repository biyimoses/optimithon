from __future__ import print_function


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

    def iterate(self, **kwargs):
        self.STEP += 1

    def terminate(self, **kwargs):
        if self.STEP >= self.MaxIteration:
            self.Terminate = True
            self.termination_message = "Maximum iteration reached."
        self.Terminate = True
        return self.Terminate


class Base(object):
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

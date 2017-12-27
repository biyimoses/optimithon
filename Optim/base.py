from __future__ import print_function


class OptimTemplate(object):
    def __init__(self, obj, **kwargs):
        from numpy import array
        self.MaxIteration = 100
        self.ErrorTolerance = 1e-10
        self.STEP = 0
        self.Success = False
        self.Terminate = False
        self.objective = None
        self.constraints = None
        self.iteration_message = None
        self.termination_message = None
        self.objective = obj
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
        _optimizer = kwargs.pop('method', OptimTemplate)
        self.optimizer = _optimizer(obj, **kwargs)

    def __call__(self, *args, **kwargs):
        # Iterate:
        while not self.optimizer.terminate(**kwargs):
            self.optimizer.iterate(**kwargs)
            # Prompt the iteration message:
            if self.Verbose:
                if self.optimizer.iteration_message is not None:
                    print(self.optimizer.iteration_message)
        # Prompt termination message:
        if self.Verbose:
            if self.optimizer.termination_message is not None:
                print(self.optimizer.termination_message)

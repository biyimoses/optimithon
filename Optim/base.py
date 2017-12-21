from __future__ import print_function


class OptimTemplate(object):
    def __init__(self):
        self.MaxIteration = 100
        self.ErrorTolerance = 1e-6
        self.STEP = 0
        self.Success = False
        self.Terminate = False
        self.objective = None
        self.constraints = None
        self.iteration_message = None
        self.termination_message = None

    def iterate(self, **kwargs):
        self.STEP += 1

    def terminate(self, **kwargs):
        if self.STEP >= self.MaxIteration:
            self.Terminate = True
            self.termination_message = "Maximum iteration reached."
        self.Terminate = True
        return self.Terminate


class Base(object):
    def __init__(self, **kwargs):
        self.x0 = None
        self.objective = lambda x: 0
        self.constraints = []
        self.Verbose = True
        _optimizer = kwargs.pop('method', OptimTemplate)
        self.optimizer = _optimizer(**kwargs)

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

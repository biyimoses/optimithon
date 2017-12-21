class OptimTemplate(object):
    def __init__(self):
        self.MaxIteration = 100
        self.ErrorTolerance = 1e-6
        self.Terminate = False

    def iterate(self):
        pass

    def terminate(self):
        self.Terminate = True
        return True


class Base(object):
    def __init__(self):
        self.x0 = None
        self.objective = lambda x: 0
        self.constraints = []
        self.STEP = 0

"""
This module provides descriptive exception for the other modules.
"""


class OptimGenericError(Exception):
    def __init__(self, *args):
        super(OptimGenericError, self).__init__(*args)


class DiffEror(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)


class MaxIterations(Exception):
    def __init__(self, *args):
        super(MaxIterations, self).__init__(*args)


class ValueRange(Exception):
    def __init__(self, *args):
        super(ValueRange, self).__init__(*args)


class Undeclared(NotImplementedError):
    def __init__(self, *args):
        super(Undeclared, self).__init__(*args)

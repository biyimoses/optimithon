"""
'excpt' Module
=================
This module provides descriptive exception for the other modules.
"""


class Error(Exception):
    r"""
    Generic errors that may occur in the course of a run.
    """
    def __init__(self, *args):
        super(Error, self).__init__(*args)


class DiffEror(Exception):
    r"""
    Errors that may have happened while calculating derivatives of functions.
    """
    def __init__(self, *args):
        super(Exception, self).__init__(*args)


class MaxIterations(Exception):
    r"""
    Errors caused by reaching the preset maximum number of iterations.
    """
    def __init__(self, *args):
        super(MaxIterations, self).__init__(*args)


class ValueRange(Exception):
    r"""
    Errors resulted from computations over unauthorized regions.
    """
    def __init__(self, *args):
        super(ValueRange, self).__init__(*args)


class DirectionError(Exception):
    r"""
    Handles the errors caused during the computation of a descent direction.
    """
    def __init__(self, *args):
        super(DirectionError, self).__init__(*args)


class Undeclared(NotImplementedError):
    r"""
    Raised when an undeclared function is used.
    """
    def __init__(self, *args):
        super(Undeclared, self).__init__(*args)

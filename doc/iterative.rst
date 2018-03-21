===============================
Iterative Optimization Methods
===============================

The iterative (unconstrained) optimization methods are the most popular optimization methods to approximate a (local) minimum of a
given function. Generally, an iterative method uses a patter like the following:

    + With the objective function :math:`f` and an initial guess for the minimum :math:`x=x_0`:
    + **Repeat**:
        - Find a descent direction :math:`p` at point :math:`x`,
        - Find a positive value :math:`\alpha` such that :math:`f(x+\alpha p)` is a reasonable decrease compare to :math:`f(x)`,
        - Update :math:`x=x+\alpha p`,
    + **Until** a termination criterion is satisfied.
    + **Return** :math:`x` as an approximation for a local minimum of :math:`f`.

Variations of the iterative methods focus on finding a suitable descent direction :math:`p`, as well as suitable value
for :math:`\alpha` and a termination strategy.

To determine a suitable direction some methods use first or second (or even higher orders) derivatives of :math:`f`.
Those methods that do not use derivatives are called *derivative free* methods.

The derivative base methods are implemented in ``QuasiNewton`` module.

Derivative Based Methods
============================

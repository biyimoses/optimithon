===================================================
Optimization Problem
===================================================
A typical optimization problem can be formulated as

.. math::
    \left\lbrace
        \begin{array}{lll}
            \min_x & f(x) & \\
            \textrm{subject to} & & \\
             & g_i(x)\ge0 & i=1,\dots,m\\
             & \textrm{and} & \\
             & h_j(x)=0 & j=1,\dots,k
        \end{array}
    \right.
    :label: CnsOptim

The problem :eq:`CnsOptim` is called a constrained optimization. If the constraints :math:`g_i(x)\ge0` and
:math:`h_j(x)=0` are absent then the :eq:`CnsOptim` is called an unconstrained optimization problem.

This package attempts to implement various methods to solve the unconstrained optimization problem.
Then by employing the barrier functions method, the resulted code for unconstrained problem is modified to solve the
general form of :eq:`CnsOptim`.

The user interface for both unconstrained and constrained optimization problems is the same, but some parameters are
ignored for unconstrained problems. A minimal code to solve :eq:`CnsOptim` would look like the following::

    from Optimithon import Base, QuasiNewton # import the essentials
    f = # definition of the objective function
    ineqs = [g_i for i in range(m)] # the list of inequality constraints: g_i >= 0.
    eqs = [h_j for j in range(k)] # the list of equality constraints: h_j == 0.
    OPTIM = Base(f, # the objective function (mandatory)
             ineq=ineqs, # inequality constraints
             eq = eqs, # equality constraints
             x0=x0, # an initial point, a numpy array
             )
    OPTIM() # run the optimization procedure
    print(OPTIM.solution) # show the outcome


===================================================
Iterative Optimization Methods
===================================================

The iterative (unconstrained) optimization methods are the most popular optimization methods to approximate a (local)
minimum of a given function. Generally, an iterative method uses a patter like the following:

    + With the objective function :math:`f` and an initial guess for the minimum :math:`x=x_0`:
    + **Repeat**:
        - Find a *descent direction* :math:`p_n` at point :math:`x_n`,
        - Find a positive value :math:`\alpha` such that :math:`f(x_n+\alpha p_n)` is a reasonable decrease compare to :math:`f(x_n)`,
        - Update :math:`x_{n+1}=x_{n}+\alpha p_n`,
    + **Until** a termination criterion is satisfied.
    + **Return** :math:`x_n` as an approximation for a local minimum of :math:`f`.

Variations of the iterative methods focus on finding a suitable *descent direction* :math:`p_n`, as well as suitable
value for :math:`\alpha` and a *termination strategy*.

To determine a suitable direction some methods use first or second (or even higher orders) derivatives of :math:`f`.
Those methods that do not use derivatives are called *derivative free* methods.

The derivative base methods are implemented in ``QuasiNewton`` module.

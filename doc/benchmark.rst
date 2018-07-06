===================================================
Benchmark Problems
===================================================
We employ the `QuasiNewton` class to solve a few benchmark optimization problems.
These benchmarks that are mainly taken from [MJXY]_.

.. [MJXY] M\. Jamil, Xin-She Yang, *A literature survey of benchmark functions for global optimization problems*, IJMMNO, Vol. 4(2) (2013).

Rosenbrock Function
==================================

The original Rosenbrock function is :math:`f(x, y)=(1-x)^2 + 100(y-x^2)^2`
which is a sums of squares and attains its minimum at :math:`(1, 1)`.
The global minimum is inside a long, narrow, parabolic shaped flat valley.
To find the valley is trivial. To converge to the global minimum, however,
is difficult.
The same holds for a generalized form of Rosenbrock function which is defined as:

.. math::
	f(x_1,\dots,x_n) = \sum_{i=1}^{n-1} 100(x_{i+1} - x_i^2)^2+(1-x_i)^2.

Since :math:`f` is a sum of squares, and :math:`f(1,\dots,1)=0`, the global
minimum is equal to 0. The following code optimizes the Rosenbrock function
over :math:`9-x_i^2\ge0` for :math:`i=1,\dots,9`::

    from Optimithon import Base
    from Optimithon import QuasiNewton
    from numpy import array

    NumVars = 9
    fun = lambda x: sum([100 * (x[i + 1] - x[i]**2)**2 +
                         (1 - x[i])**2 for i in range(NumVars - 1)])
    x0 = array([0 for _ in range(NumVars)])

    OPTIM = Base(fun, ineq=[lambda x: 9 - x[i]**2 for i in range(NumVars)],
                 br_func='Carrol',
                 penalty=1.e6,
                 method=QuasiNewton, x0=x0,
                 t_method='Cauchy',
                 dd_method='BFGS',
                 ls_method='Backtrack',
                 ls_bt_method='Armijo',
                 )
    OPTIM.Verbose = False
    OPTIM.MaxIteration = 1500
    OPTIM()
    print(OPTIM.solution)

The result is::

    objective: 1.55528520803e-11
    x: [ 1.0000001   1.00000013  1.00000013  1.00000005  0.99999996  0.99999993
	1.00000007  0.99999997  0.99999989]
    NumIteration: 55
    NumFuncEval: 256
    success: True
    message: Progress in objective values less than error tolerance (Cauchy condition)
    RunTime: 0.0378611087799
    Family: Quasi-Newton method
    Step Size: Backtrack
    Backtrack Stop Criterion: Armijo
    Descent Direction: BFGS
    Termination Criterion: Cauchy
    Barrier Function: Carrol
    Penalty Factor: 1000000.0

Giunta Function
==================================

Giunta is an example of continuous, differentiable, separable, scalable,
multimodal function defined by:

.. math::
    \begin{array}{lcl}
    f(x_1, x_2) & = & \frac{3}{5} + \sum_{i=1}^2[\sin(\frac{16}{15}x_i-1)\\
		& + & \sin^2(\frac{16}{15}x_i-1)\\
		& + & \frac{1}{50}\sin(4(\frac{16}{15}x_i-1))].
    \end{array}


The following code optimizes :math:`f` when :math:`1-x_i^2\ge0`::

    from Optimithon import Base
    from Optimithon import QuasiNewton
    from numpy import array, sin

    fun = lambda x: .6 + (sin((16. / 15.) * x[0] - 1) + (sin((16. / 15.) * x[0] - 1))**2 + .02 * sin(4 * ((16. / 15.) * x[0] - 1))) + (
        sin((16. / 15.) * x[1] - 1) + (sin((16. / 15.) * x[1] - 1))**2 + .02 * sin(4 * ((16. / 15.) * x[1] - 1)))
    x0 = array([0 for _ in range(2)])

    OPTIM = Base(fun, ineq=[lambda x: 1 - x[i]**2 for i in range(2)],
                 br_func='Carrol',
                 penalty=1.e6,
                 method=QuasiNewton, x0=x0,
                 t_method='Cauchy_x',
                 dd_method='BFGS',
                 ls_method='Backtrack',
                 ls_bt_method='Armijo',
                 )
    OPTIM.Verbose = False
    OPTIM.MaxIteration = 1500
    OPTIM()
    print(OPTIM.solution)

The output looks like::

    objective: 0.0644704205391
    x: [ 0.46732003  0.46731857]
    NumIteration: 8
    NumFuncEval: 20
    success: True
    message: The progress in values of points is less than error tolerance (0.000000)
    RunTime: 0.0011510848999
    Family: Quasi-Newton method
    Step Size: Backtrack
    Backtrack Stop Criterion: Armijo
    Descent Direction: BFGS
    Termination Criterion: Cauchy_x
    Barrier Function: Carrol
    Penalty Factor: 1000000.0

Parsopoulos Function
==================================

Parsopoulos is defined as :math:`f(x,y)=\cos^2(x)+\sin^2(y)`.
The following code computes its minimum where :math:`-5\leq x,y\leq5`::

    from Optimithon import Base
    from Optimithon import QuasiNewton
    from numpy import array, sin, cos

    fun = lambda x: cos(x[0])**2 + sin(x[1])**2
    x0 = array((1., -2.))

    OPTIM = Base(fun, ineq=[lambda x: 25. - x[j]**2 for j in range(2)],
                 br_func='Carrol',
                 penalty=1.e6,
                 method=QuasiNewton, x0=x0,
                 t_method='Cauchy_x',
                 dd_method='BFGS',
                 ls_method='BarzilaiBorwein',
                 )
    OPTIM.Verbose = False
    OPTIM.MaxIteration = 1500
    OPTIM()
    print(OPTIM.solution)

The solution is the following::

    objective: 7.48150734385e-16
    x: [ 1.57079633 -3.14159263]
    NumIteration: 33
    NumFuncEval: 36
    success: True
    message: The progress in values of points is less than error tolerance (0.000000)
    RunTime: 0.00488901138306
    Family: Quasi-Newton method
    Step Size: BarzilaiBorwein
    Descent Direction: BFGS
    Termination Criterion: Cauchy_x
    Barrier Function: Carrol
    Penalty Factor: 1000000.0

Shubert Function
==================================

Shubert function is defined by:

.. math::
    f(x_1,\dots,x_n) = \prod_{i=1}^n\left(\sum_{j=1}^5\cos((j+1)x_i+i)\right).

It is a continuous, differentiable, separable, non-scalable, multimodal function.
The following code compares the result of five optimizers when :math:`-10\leq x_i\leq10`
and :math:`n=2`::

    from Optimithon import Base
    from Optimithon import QuasiNewton
    from numpy import array, cos

    fun = lambda x: sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * \
        sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])
    x0 = array((1., -1.))

    OPTIM = Base(fun, ineq=[lambda x: 100. - x[i]**2 for i in range(2)],
                 br_func='Carrol',
                 penalty=1.e6,
                 method=QuasiNewton, x0=x0,
                 t_method='Cauchy',
                 dd_method='Gradient',
                 ls_method='Backtrack',
                 ls_bt_method='Armijo',
                 )
    OPTIM.Verbose = False
    OPTIM.MaxIteration = 1500
    OPTIM()
    print(OPTIM.solution)

which results in::

    objective: -18.09556507
    x: [-7.06139727 -1.47136939]
    NumIteration: 51
    NumFuncEval: 1021
    success: True
    message: Progress in objective values less than error tolerance (Cauchy condition)
    RunTime: 2.48312807083
    Family: Quasi-Newton method
    Step Size: Backtrack
    Backtrack Stop Criterion: Armijo
    Descent Direction: Gradient
    Termination Criterion: Cauchy
    Barrier Function: Carrol
    Penalty Factor: 1000000.0

McCormick Function
==================================
McCormick function is defined by

.. math::
    f(x, y) = \sin(x+y) + (x-y)^2-1.5x+2.5y+1.

Attains its minimum at :math:`f(-.54719, -1.54719)\approx-1.9133`::

    from Optimithon import Base
    from Optimithon import QuasiNewton
    from numpy import array, sin
    from scipy.optimize import minimize

    fun = lambda x: sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1.
    x0 = array((0., 0.))

    OPTIM = Base(fun,
                 method=QuasiNewton, x0=x0,  # max_lngth=100.,
                 t_method='Cauchy_x',  # 'Cauchy_x', 'ZeroGradient',
                 dd_method='BFGS',
                 # 'Newton', 'SR1', 'HestenesStiefel', 'PolakRibiere', 'FletcherReeves', 'Gradient', 'DFP', 'BFGS', 'Broyden', 'DaiYuan'
                 ls_method='Backtrack',  # 'BarzilaiBorwein', 'Backtrack',
                 ls_bt_method='Armijo',  # 'Armijo', 'Goldstein', 'Wolfe', 'BinarySearch'
                 )
    OPTIM.Verbose = False
    OPTIM.MaxIteration = 1500
    OPTIM()
    print(OPTIM.solution)

The output will be::

    objective: -1.91322295498
    x: [-0.54719755 -1.54719755]
    NumIteration: 9
    NumFuncEval: 23
    success: True
    message: The progress in values of points is less than error tolerance (0.000000)
    RunTime: 0.000735998153687
    Family: Quasi-Newton method
    Step Size: Backtrack
    Backtrack Stop Criterion: Armijo
    Descent Direction: BFGS
    Termination Criterion: Cauchy_x
    Barrier Function: Carrol
    Penalty Factor: 100000.0


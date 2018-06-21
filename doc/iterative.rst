===============================
Iterative Optimization Methods
===============================

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

===================================================
Derivative Based Methods
===================================================
Derivative Based Methods are a class of iterative optimization methods that calculate a descent direction using gradient
and/or hessian of the objective function.

---------------------------------------------------
Descent Direction
---------------------------------------------------
The following is the list of implemented methods to find a descent direction.

Gradient descent direction
---------------------------------------------------
This method chooses the backward gradient direction to achieve the direction which results in steepest decrease in the
values of the objective function:

.. math::
    p_n=\nabla f(x_n)

Newton Conjugate Gradient method
---------------------------------------------------
The Newton conjugate gradient method uses the followin descent direction:

.. math::
    p_{n+1}=\nabla^2f(x_n)^{-1}\nabla f(x_n),

where :math:`\nabla^2f(x_n)` is the Hessian of :math:`f` at :math:`x_n`.

Fletcher-Reeves method
---------------------------------------------------
The direction suggested by R. Fletcher and C. M. Reeves in 1964. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\nabla f(x_n)^T\nabla f(x_n)}{\|\nabla f(x_{n-1})\|^2}p_{n-1}-\nabla f(x_n).

Polak–Ribière method
---------------------------------------------------
Suggested by E. Polak and G. Ribière in 1969. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\nabla f(x_n)^T(\nabla f(x_n)-\nabla f(x_{n-1}))}{\|\nabla f(x_{n-1})\|^2}p_{n-1}-\nabla f(x_n).

Hestenes-Stiefel method
---------------------------------------------------
Suggested by M. R. Hestenes and E. Stiefel in 1953. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\nabla f(x_n)^T(\nabla f(x_n)-\nabla f(x_{n-1}))}{(\nabla f(x_n)-\nabla f(x_{n-1}))^Tp_{n-1}}p_{n-1}-\nabla f(x_n).

Dai-Yuan method
---------------------------------------------------
Suggested by Y.-H. Dai and Y. Yuan in 1999. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\|\nabla f(x_n)\|^2}{(\nabla f(x_n)-\nabla f(x_{n-1}))p_{n-1}}p_{n-1}-\nabla f(x_n).

Davidon-Fletcher-Powell method
---------------------------------------------------
Let :math:`H_0=\nabla^2f(x_0)^{-1}` and

.. math::
    \begin{array}{lcl}
    H_n & = & H_{n-1}+\frac{(x_n - x_{n-1})^T(x_n - x_{n-1})}{(x_n - x_{n-1})^T(\nabla f(x_n)-\nabla f(x_{n-1}))}\\
     & - & \frac{H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))(\nabla f(x_n)-\nabla f(x_{n-1}))^TH_{n-1}}{(\nabla f(x_n)-
    \nabla f(x_{n-1}))^TH_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))},
    \end{array}

then :math:`p_n=-H_n\nabla f(x_n)`.

Broyden-Fletcher-Goldfarb-Shanno method
---------------------------------------------------
Let :math:`H_0=\nabla^2f(x_0)^{-1}` and

.. math::
    \begin{array}{lcl}
    H_n & = & \left(I-\frac{(x_n - x_{n-1})(\nabla f(x_n)-\nabla f(x_{n-1}))^T}{(\nabla f(x_n)-\nabla f(x_{n-1}))^T(x_n - x_{n-1})}\right)\\
     & \times & H_{n-1}\\
     & \times & \left(I-\frac{(\nabla f(x_n)-\nabla f(x_{n-1}))(x_n - x_{n-1})^T}{(\nabla f(x_n)-\nabla f(x_{n-1}))^T(x_n - x_{n-1})}\right)\\
     & + & \frac{(x_n - x_{n-1})(x_n - x_{n-1})^T}{(\nabla f(x_n)-\nabla f(x_{n-1}))^T(x_n - x_{n-1})},
    \end{array}

then :math:`p_n=-H_n\nabla f(x_n)`.

Broyden’s method
---------------------------------------------------
Let :math:`H_0=\nabla^2f(x_0)^{-1}` and

.. math::
    \begin{array}{lcl}
    H_n & = & H_{n-1}\\
     & + & \frac{((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-f(x_{n-1})))(x_n- x_{n-1})^TH_{n-1}}
    {(x_n- x_{n-1})^TH_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))},
    \end{array}

then :math:`p_n=-H_n\nabla f(x_n)`.

Symmetric Rank-One (SR1) method
---------------------------------------------------
Let :math:`H_0=\nabla^2f(x_0)^{-1}` and

.. math::
    \begin{array}{lcl}
    H_n & = & H_{n-1}\\
     & + & \frac{\((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))^T}
     {((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))^T(\nabla f(x_n)-\nabla f(x_{n-1})},
    \end{array}

then :math:`p_n=-H_n\nabla f(x_n)`.

---------------------------------------------------
Line Search methods
---------------------------------------------------
In every iteration, beside finding a descent direction, the algorithm also requires the magnitude of the descent,
denoted by :math:`\alpha` in the algorithm. One popular method to find :math:`\alpha` is called line search.
The following is the list of line search methods implemented.

Barzilai-Borwein method
---------------------------------------------------
The length of the descent direction suggested by Barzilai-Borwein method is calculated with the following formula:

.. math::
    \alpha=\frac{(x_n- x_{n-1})(\nabla f(x_n)-\nabla f(x_{n-1}))^T}{\|\nabla f(x_n)-\nabla f(x_{n-1})\|^2}.

Backtrack line search method
---------------------------------------------------
Backtrack line search is a generic algorithm relying in various conditions to approximate a suitable magnitude for the
descent direction.

Starting with a maximum candidate step size value :math:`\alpha_0>0`, using search control parameters
:math:`\tau\in(0,1)` and :math:`c\in(0,1)`, the backtracking line search algorithm can be expressed as follows:

    + Set :math:`t=-cm` and iteration counter :math:`j=0`.
    + Until a condition :math:`\dagger` is satisfied, repeatedly increment :math:`j` and set :math:`\alpha_j=\tau\alpha_{j-1}`.
    + Return :math:`\alpha_j` as the solution.

The :math:`\dagger` condition is usually one of the following:
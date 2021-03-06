===================================================
Derivative Based Methods
===================================================
Derivative Based Methods are a class of iterative optimization methods that calculate a descent direction using gradient
and/or hessian of the objective function.

For more details on the methods introduced in this chapter refer to [JNSJW]_.

.. [JNSJW] J\. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd  ed., Springer, New York, NY, USA (2006).

---------------------------------------------------
Descent Direction
---------------------------------------------------
The following is the list of implemented methods to find a descent direction.

Gradient descent direction
---------------------------------------------------
This method chooses the backward gradient direction to achieve the direction which results in steepest decrease in the
values of the objective function:

.. math::
    p_n=-\nabla f(x_n)

Newton Conjugate Gradient method
---------------------------------------------------
The Newton conjugate gradient method uses the followin descent direction:

.. math::
    p_{n+1}=\nabla^2f(x_n)^{-1}\nabla f(x_n),

where :math:`\nabla^2f(x_n)` is the Hessian of :math:`f` at :math:`x_n`.

Fletcher-Reeves method
---------------------------------------------------
The direction suggested by R. Fletcher and C. M. Reeves in 1964 [RFCMR]_. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\nabla f(x_n)^T\nabla f(x_n)}{\|\nabla f(x_{n-1})\|^2}p_{n-1}-\nabla f(x_n).

.. [RFCMR] R\. Fletcher and C. M. Reeves, *Function minimization by conjugate gradients*, Comput. J. 7 (1964), 149–154.

Polak–Ribiere method
---------------------------------------------------
Suggested by E. Polak and G. Ribiere in 1969 [EPGR]_. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\nabla f(x_n)^T(\nabla f(x_n)-\nabla f(x_{n-1}))}{\|\nabla f(x_{n-1})\|^2}p_{n-1}-\nabla f(x_n).

.. [EPGR] E\. Polak and G. Ribiere, *Note sur la convergence de directions conjuguee*, Rev. Francaise Informat Recherche Operationelle, 3e Annee 16 (1969), 35–43.

Hestenes-Stiefel method
---------------------------------------------------
Suggested by M. R. Hestenes and E. Stiefel in 1953 [MRHES]_. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\nabla f(x_n)^T(\nabla f(x_n)-\nabla f(x_{n-1}))}{(\nabla f(x_n)-\nabla f(x_{n-1}))^Tp_{n-1}}p_{n-1}-\nabla f(x_n).

.. [MRHES] M\. R. Hestenes and E. Stiefel, *Methods of conjugate gradients for solving linear systems*, J. Research Nat. Bur. Standards 49 (1952), 409–436 (1953).

Dai-Yuan method
---------------------------------------------------
Suggested by Y.-H. Dai and Y. Yuan in 1999 [YHDYY]_. Let :math:`p_0=-\nabla f(x_0)` and

.. math::
    p_n=\frac{\|\nabla f(x_n)\|^2}{(\nabla f(x_n)-\nabla f(x_{n-1}))p_{n-1}}p_{n-1}-\nabla f(x_n).

.. [YHDYY] Y\.-H. Dai and Y. Yuan, *A nonlinear conjugate gradient method with a strong global convergence property*, SIAM J. Optim. 10 (1999), no. 1, 177–182.

Davidon-Fletcher-Powell method
---------------------------------------------------
Let :math:`H_0=\nabla^2f(x_0)^{-1}` and

.. math::
    \begin{array}{lcl}
    H_n & = & H_{n-1}+\frac{(x_n - x_{n-1})^T(x_n - x_{n-1})}{(x_n - x_{n-1})^T(\nabla f(x_n)-\nabla f(x_{n-1}))}\\
     & - & \frac{H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))(\nabla f(x_n)-\nabla f(x_{n-1}))^TH_{n-1}}{(\nabla f(x_n)-
    \nabla f(x_{n-1}))^TH_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))},
    \end{array}

then :math:`p_n=-H_n\nabla f(x_n)` (see [WCD]_ and [RF]_).

.. [WCD] W\. C. Davidon, *Variable metric method for minimization*, SIAM Journal on Optimization, 1: 1–17 (1991).
.. [RF] R\. Fletcher, *Practical methods of optimization* (2nd ed.), New York: John Wiley & Sons (1987).

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

then :math:`p_n=-H_n\nabla f(x_n)` (see [RF]_).

Broyden’s method
---------------------------------------------------
Let :math:`H_0=\nabla^2f(x_0)^{-1}` and

.. math::
    \begin{array}{lcl}
    H_n & = & H_{n-1}\\
     & + & \frac{((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-f(x_{n-1})))(x_n- x_{n-1})^TH_{n-1}}
    {(x_n- x_{n-1})^TH_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))},
    \end{array}

then :math:`p_n=-H_n\nabla f(x_n)` (see [CGB]_).

.. [CGB] C\. G. Broyden, *A Class of Methods for Solving Nonlinear Simultaneous Equations*. Math. of Comput. AMS. 19 (92): 577–593 (1965).

Symmetric Rank-One (SR1) method
---------------------------------------------------
Let :math:`H_0=\nabla^2f(x_0)^{-1}` and

.. math::
    \begin{array}{lcl}
    H_n & = & H_{n-1}\\
     & + & \frac{((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))^T}
     {((x_n- x_{n-1})-H_{n-1}(\nabla f(x_n)-\nabla f(x_{n-1}))^T(\nabla f(x_n)-\nabla f(x_{n-1}))},
    \end{array}

then :math:`p_n=-H_n\nabla f(x_n)` (see [RHB]_).

.. [RHB] R\. H. Byrd *Analysis of a Symmetric Rank-One Trust Region Method*, SIAM J. Optim 6(4) (1996).

---------------------------------------------------
Line Search methods
---------------------------------------------------
In every iteration, beside finding a descent direction, the algorithm also requires the magnitude of the descent,
denoted by :math:`\alpha` in the algorithm. One popular method to find :math:`\alpha` is called line search.
The following is the list of line search methods implemented.

Barzilai-Borwein method
---------------------------------------------------
The length of the descent direction suggested by Barzilai-Borwein method [JBJMB]_ is calculated with the following
formula:

.. math::
    \alpha=\frac{(x_n- x_{n-1})(\nabla f(x_n)-\nabla f(x_{n-1}))^T}{\|\nabla f(x_n)-\nabla f(x_{n-1})\|^2}.

.. [JBJMB] J\. Barzilai, J. M. Borwein. *Two-point step size gradient methods*, IMA J. Numerical Analysis, 8(1):141–148 (1988).

Backtrack line search method
---------------------------------------------------
Backtrack line search is a generic algorithm relying in various conditions to approximate a suitable magnitude for the
descent direction [JNSJW]_.

Starting with a maximum candidate step size value :math:`\alpha_0>0`, using search control parameters
:math:`\tau\in(0,1)` and :math:`c\in(0,1)`, the backtracking line search algorithm can be expressed as follows:

    + Set :math:`t=-cp_n\cdot\nabla f(x_n)` and iteration counter :math:`j=0`.
    + Until a condition :math:`\dagger(\alpha_j, t)` is satisfied, repeatedly increment :math:`j` and set :math:`\alpha_j=\tau\alpha_{j-1}`.
    + Return :math:`\alpha_j` as the solution.

The :math:`\dagger` condition is usually one of the following:

    + **Wolfe condition:** :math:`p_n\cdot\nabla f(x_n+\alpha_j p_n)\ge t`
    + **Armijo condition:** :math:`\alpha_jt\ge f(x_n+\alpha_jp_n)-f(x_n)`
    + **Goldstein condition:**
        - :math:`f(x_n)+(1-c)\alpha_jt\leq f(x_n+\alpha_jp_n)` and
        - :math:`f(x_n+\alpha_jp_n)\leq f(x_n)+\alpha_jt`
    + **Strong Wolfe condition:**
        - :math:`f(x_n+\alpha_jp_n)\leq f(x_n)+c_1\alpha_jt` and
        - :math:`|p_n\nabla f(x_n+\alpha_jp_n)|\leq c_2|t|` for :math:`0<c_1<c_2<1`
    + **Binary Search method**: :math:`f(x_n+\alpha_jp_n)<f(x_n)`

---------------------------------------------------
Termination criterion
---------------------------------------------------
At the end of every iteration a termination criterion is evaluated to decide continuation or break of the loop.
The following is a list of implemented methods:

Cauchy condition
---------------------------------------------------
Given the sequence of calculated points :math:`(x_n)`, this condition checks whether the values of the objective are
making enough progress or reached a limit point. In symbols, for :math:`\varepsilon>0`,

.. math::
    |f(x_n)-f(x_{n-1})|<\varepsilon.

Cauchy_x condition
---------------------------------------------------
Given the sequence of calculated points :math:`(x_n)`, this condition checks whether this sequence is making enough
progress or reached an approximate limit point. In symbols, for :math:`\varepsilon>0`,

.. math::
    \|x_n - x_{n+1}\|<\varepsilon.

ZeroGradient condition
---------------------------------------------------
This condition checks the size of gradient vector at each point found at the end of iteration. If the gradient vector
is close enough to zero, then it means that the values of the objective will not make significant progress at any
direction. In symbols, for :math:`\varepsilon>0`,

.. math::
    \|\nabla f(x_n)\|<\varepsilon.

Note that this condition may not be suitable to solve constrained optimization problems.

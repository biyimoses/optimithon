===================================================
Constrained Optimization
===================================================
It is possible to transform a constrained optimization problem :eq:`CnsOptim`  to an unconstrained one with some
conditions on the solutions. A popular method to do so is known as barrier function where the objective function is
modified to penalize the search method if any of the constraints is violated.

---------------------------------------------------
Barrier Function Method
---------------------------------------------------
Let :math:`\phi` be a function that takes over relatively small values (compare to values of :math:`f`) over positive
reals and big values for negative ones. Then the function :math:`f(x)+\sum_{i=1}^m\phi(g_i(x))` is fairly large values
if :math:`x` is outside of the feasibility region. Therefore, it is likely that a search method tends to focus on the
values inside the feasibility region. Similarly, let :math:`\psi` be a function that returns large positive values
apart from :math:`0`. Then :math:`f(x)+\sum_{j=1}^k\psi(h_j(x))` is large outside of the feasibility region and rather
small on the feasibility region.

Since the values of continuous functions changes gradually, it seems implausible to be able to choose a function that
successfully accomplish the above task. So, we employ an increasing sequence of positive numbers :math:`(\sigma_n)`
that approaches to :math:`+\infty` and find the optimum value for the function

.. math::
    \Lambda_n(x)=f(x)+\frac{1}{\sigma_n}\sum_{i=1}^m\phi(g_i(x))+\sigma_n\sum_{j=1}^k\psi(h_j(x)),

then the optimum values of :math:`\Lambda_n` approaches the optimum value of :math:`f` inside the feasibility region.
Note that if the optimum value is located on the boundary of the feasibility region, this method would only produce an
approximate interior substitute. This is why the method is referred to as interior point method.

The following options for the barrier function are implemented:

    + For the inequality conditions:
        - `Carrol`: is the standard barrier function defined by :math:`-\frac{1}{g_i(x)}`
        - `Logarithmic`: is the standard barrier function defined by :math:`-\log(g_i(x))`
        - `Expn`: is the standard barrier function defined by :math:`e^{-g_i(x)+\epsilon}`
    + For the equality condition:
        - `Courant`: function which is simply defined by :math:`h_j^2(x)`.

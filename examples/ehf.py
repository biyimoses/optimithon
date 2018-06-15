import os
from sympy import *
from sympy.abc import x
from sympy.utilities.lambdify import implemented_function
import numpy as np
from gurobipy import *
from pyProximation import Measure, OrthSystem, Graphics


def polyGen(meanList):
    polyFile = open("polySig.txt", "w")
    x = Symbol('x')

    # define ReLU function symbolically and numerically
    ReLU = implemented_function(Function('ReLU'), lambda a: max(0, a))
    absReLU = implemented_function(Function('ReLU'), lambda a: (abs(a) + a) / 2)
    Sigmoid = implemented_function(Function('sigmoid'), lambda a: 1 / (1 + exp(-a)))
    Inverse = implemented_function(Function('inverse'), lambda a: 1 / (a + 1000))
    tanh = implemented_function(Function('tanh'), lambda a: tanh(a))
    f = Sigmoid(x)
    g = ReLU(x)

    error = 0.1  # Intentional added error (negligible for large values of `l`)
    counter = 1

    # meanList = set(meanList)
    # meanList = map(int,meanList)

    for l in meanList:
        D = [((-1) * l, l)]
        w = lambda x: 1. / sqrt(1. - (x / l) ** 2)
        mu = lambda x: exp(-1. / (1e-5 + (x / l) ** 8))
        # Half the length of a symmetric interval about 0
        # l = 5000 # Half the length of a symmetric interval about 0
        polyFile.write("#Interval = [" + str(-l) + "," + str(l) + "]\n")

        for precision in range(2, 3):
            ring = RealField(precision)
            polyFile.write("#presicion = " + str(precision) + "\n")

            for n in range(2, 4, 2):
                str1 = ""

                M = Measure(D, w)
                N = Measure(D, mu)
                S = OrthSystem([x], D, 'sympy')
                T = OrthSystem([x], D, 'sympy')
                U = OrthSystem([x], D, 'sympy')
                # link the measure to S
                S.SetMeasure(M)
                U.SetMeasure(N)
                # set B = {1, x, x^2, ..., x^n}
                B = S.PolyBasis(n)
                # link B to S
                S.Basis(B)
                T.Basis(B)
                U.Basis(B)
                # generate the orthonormal basis
                S.FormBasis()
                T.FormBasis()
                U.FormBasis()
                # number of elements in the basis
                m = len(S.OrthBase)
                # set f(x) = ReLU(x)

                # extract the coefficients
                Coeffs1 = S.Series(f)
                Coeffs2 = T.Series(f)
                Coeffs3 = U.Series(f)
                Coeffs3[0] = int(Coeffs3[0] * 100)
                Coeffs3[1] = int(Coeffs3[1] * 100)
                # Coeffs3 = np.array(Coeffs3)

                # print Coeffs3
                # Coeffs3 = Coeffs3.astype(int)
                # # print Coeffs3
                # form the approximation
                f_app1 = sum([S.OrthBase[i] * Coeffs1[i] for i in range(m)])
                f_app2 = sum([T.OrthBase[i] * Coeffs2[i] for i in range(m)])
                f_app3 = sum([U.OrthBase[i] * Coeffs3[i] for i in range(m)])

                polyFile.write("#Chebyshev-l" + (str(l)) + "-d" + str(n) + ":\n")
                print f_app3
                integ_f_app3 = f_app3.integrate(x)
                print integ_f_app3

                nf = lambdify(x, g - integ_f_app3)
                polyFile.write("@@Norm-2 of the approximation error Chebyshev = %f\n" % (N.norm(2, nf)))

                str1 = "def polyReLUInteg" + (str(counter)) + "(x):\n    return "
                polyFile.write(str1)
                str1 = str(integ_f_app3) + "\n\n"
                polyFile.write(str1)

                degreePoly = degree(integ_f_app3)
                Q = [[0 for i in range(degreePoly + 1)] for i in range(degreePoly + 1)]

                # print diff(mu(x), x)

                for i in range(degreePoly + 1):
                    for j in range(degreePoly + 1):
                        f1 = lambda x: x ** (i + j)
                        Q[i][j] = 2 * N.integral(f1)

                # print Q

                c = []

                for i in range(degreePoly + 1):
                    f1 = lambdify(x, integ_f_app3)
                    f2 = lambda x: x ** i * f1(x)
                    m = N.integral(f2)
                    c.append(-2 * m)

                # print c

                counter = counter + 1

                # print "Chebyshev: ", f_app1
                # print "Legendre: ", f_app2
                # print "The other one: ", f_app3
                # G = Graphics('sympy', numpoints=100)
                # I = 1000
                # # G.Plot2D(w(x), (x, -1, 1), color='red', legend='Chebyshev')
                # # G.Plot2D(mu(x), (x, -10, 10), color='green', legend='Legendre')
                # G.Plot2D(integ_f_app3, (x, (-1)*I, I), color='pink', legend="The other one")
                # integ_f_app2 = (2*x**2 + 50*x)/10000
                # integ_f_app2 = ((2*x**2 + 50*x)-10000)**2+((2*x**2 + 50*x)-1000)
                # G.Plot2D(integ_f_app2, (x, (-1)*I, I), color='green', legend="integer One")
                # G.Plot2D(g, (x, (-1)*I, I), color='blue', legend='ReLU')
                # G.SetTitle("Approximations of degree %d"%n)
                # G.save('ReLU'+str(n)+'.png')

    return Q, c


def MIQPSolver(Q, C):
    # !/usr/bin/python

    # Copyright 2017, Gurobi Optimization, Inc.

    # This eq0ample formulates and solves the following simple QP model:
    #  minimize
    #      q0^2 + q0*q1 + q1^2 + q1*q2 + q2^2 + 2 q0
    #  subject to
    #      q0 + 2 q1 + 3 q2 >= 4
    #      q0 +   q1       >= 1
    #
    # It solves it once as a continuous model, and once as an integer model.

    from gurobipy import *

    # Create a new model
    m = Model("qp")

    # Create variables
    q0 = m.addVar(vtype=GRB.INTEGER, name="q0")
    q1 = m.addVar(vtype=GRB.INTEGER, name="q1")
    q2 = m.addVar(vtype=GRB.INTEGER, name="q2")

    # q0 = m.addVar( name="q0")
    # q1 = m.addVar( name="q1")
    # q2 = m.addVar( name="q2")

    # Set objective: q0^2 + q0*q1 + q1^2 + q1*q2 + q2^2 + 2 q0
    obj = (-1) * (0.5 * (q0 * q0 * Q[0][0] + q0 * q1 * Q[1][0] + \
                         q0 * q2 * Q[2][0] + q0 * q1 * Q[0][1] + \
                         q1 * q1 * Q[1][1] + q1 * q2 * Q[2][2] + \
                         q0 * q2 * Q[0][2] + q1 * q2 * Q[1][2] + \
                         q2 * q2 * Q[2][2]) + q0 * C[0] + q1 * C[1] + q2 * C[2])

    m.setObjective(obj, GRB.MAXIMIZE)

    # Add constraint: q0 + 2 q1 + 3 q2 <= 4
    # m.addConstr(q0 + 2 * q1 + 3 * q2 >= 4, "c0")

    # Add constraint: q0 + q1 >= 1
    # m.addConstr(q0 + q1 >= 1, "c1")

    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % obj.getValue())

    q0.vType = GRB.INTEGER
    q1.vType = GRB.INTEGER
    q2.vType = GRB.INTEGER

    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % obj.getValue())


Q, C = polyGen([1])
# print Q
# print C
MIQPSolver(Q, C)
from sympy import Symbol, solve, lambdify, Matrix
import numpy as numpy
import matplotlib.pyplot as matplot
from scipy.integrate import odeint

#инициализация символьных переменных
km1 = Symbol('km1', Positive=True)
km2 = Symbol('km2', Positive=True)
k3_0 = Symbol('k3_0', Positive=True)
alpha = Symbol('alpha', Positive=True)
k1 = Symbol('k1', Positive=True)
k2 = Symbol('k2', Positive=True)
x = Symbol("x", Positive=True)
y = Symbol("y", Positive=True)

#инициализация констант
alpha_value = 16.0
km1_value = 0.01
km2_value = 0.01
k3_0_value = 10.0

alpha_range = [10,15,18,20,25]
k3_range = [1,5,10,50,100]

#задаем систему уравнений
eq1 = k1 * (1 - x - y) - km1 * x- x * y * k3_0 * (1 - y) ** alpha
eq2 = k2 * (1 - x - y) ** 2 - km2 * y ** 2 - x * y * k3_0 * (1 - y) ** alpha

#решаем систему уравнений
solution = solve([eq1, eq2], x, k2)
x_Solution = solution[0][0]
k2_Solution = solution[0][1]

#ищем промежуточные значения
A = Matrix([eq1, eq2])
var_vector = Matrix([x, y])
jac_A = A.jacobian(var_vector)
det_A = jac_A.det()
trace_A = jac_A.trace()

Y = numpy.arange(0, 1, 1e-3)

#двухпараметрический анализ k1, k2
def Analysis_K1_K2():
    #подсчет линий кратности
    # подсчет 
    k2_Trace_Solution = solve(trace_A.subs(x,x_Solution),k2)[0]
    k1_Join_Solution = solve(k2_Trace_Solution - k2_Solution, k1)[0]
    k2_Join_Solotion = k2_Trace_Solution.subs(k1, k1_Join_Solution)
    k1_Trace_y = lambdify((y,km1,km2,k3_0,alpha),k1_Join_Solution,'numpy')
    k2_Trace_y = lambdify((y,km1,km2,k3_0,alpha),k2_Join_Solotion,'numpy')
    #подсчет

    return


def Solve_System(init, k1val, k1mval, k2val, k2mval, k3_0val,alphaval, dt, iterations):

    f1 = lambdify((x, y, k1, km1, k3_0,alpha), eq1)
    f2 = lambdify((x, y, k2, km2, k3_0,alpha), eq2)

    def rhs(xy, times):
        return [f1(xy[0], xy[1], k1val, k1mval, k3_0val,alphaval), f2(xy[0], xy[1],k2val, k2mval, k3_0val,alphaval)]

    times = numpy.arange(iterations) * dt
    return odeint(rhs, init, times), times

#отрисовка колебаний
def Auto_Col():
    res, times = Solve_System([0.38, 0.22], 0.03, 0.01, 0.05, 0.01, 10, 16, 1e-2, 1e6)

    ax = matplot.subplot(211)
    matplot.plot(times, res[:, 0])
    matplot.setp(ax.get_xticklabels(), visible=False)
    matplot.ylabel('x')
    matplot.grid()
    matplot.subplot(212, sharex=ax)
    matplot.plot(times, res[:, 1], color='red')
    matplot.xlabel('t')
    matplot.ylabel('y')
    matplot.grid()
    matplot.show()
    return

#отрисовка в фазовой плоскости
def Stream_Plot(k1val, k1mval, k2val, k2mval, k3_0val,alphaval):
    f1 = lambdify((x, y, k1, km1, k3_0, alpha), eq1)
    f2 = lambdify((x, y, k2, km2, k3_0, alpha), eq2)
    Y, X = numpy.mgrid[0:.5:1000j, 0:1:2000j]
    U = f1(X, Y, k1val, k1mval, k3_0val,alphaval)
    V = f2(X, Y, k2val, k2mval, k3_0val,alphaval)
    velocity = numpy.sqrt(U*U + V*V)
    matplot.streamplot(X, Y, U, V, density = [2.5, 0.8], color=velocity)
    matplot.xlabel('x')
    matplot.ylabel('y')
    matplot.show()

Stream_Plot(0.03, 0.01, 0.05, 0.01, 10,16)
Auto_Col()





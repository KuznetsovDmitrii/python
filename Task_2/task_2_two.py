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

#задаем систему уравнений
eq1 = k1 * (1 - x - y) - km1 * x - x * y * k3_0 * (1 - y) ** alpha
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
    # линии нейтральности
    # подсчет
    k2_Trace_Solution = solve(trace_A.subs(x, x_Solution), k2)[0]
    k1_Join_Trace_Solution = solve(k2_Trace_Solution - k2_Solution, k1)[0]
    k2_Join_Trace_Solotion = k2_Trace_Solution.subs(k1, k1_Join_Trace_Solution)
    k1_Trace_y = lambdify((y, km1, km2, k3_0, alpha), k1_Join_Trace_Solution, 'numpy')
    k2_Trace_y = lambdify((y, km1, km2, k3_0, alpha), k2_Join_Trace_Solotion, 'numpy')

    # линии кратности
    # подсчет
    k2_Det_Solution = solve(det_A.subs(x, x_Solution), k2)[0]
    k1_Join_Det_Solution = solve((k2_Det_Solution - k2_Solution), k1)[0]
    k2_Join_Det_Solotion = k2_Solution.subs(k1, k1_Join_Det_Solution)
    k1_Det_y = lambdify((y, km1, km2, k3_0, alpha), k1_Join_Det_Solution, 'numpy')
    k2_Det_y = lambdify((y, km1, km2, k3_0, alpha), k2_Join_Det_Solotion, 'numpy')

    # отрисовка линий кратности и нейтральности
    matplot.plot(k2_Trace_y(Y, km1_value, km2_value, k3_0_value, alpha_value),
                 k1_Trace_y(Y, km1_value, km2_value, k3_0_value, alpha_value),
                 linestyle='--', linewidth=1.5, label='neutral')
    matplot.plot(k2_Det_y(Y, km1_value, km2_value, k3_0_value, alpha_value),
                 k1_Det_y(Y, km1_value, km2_value, k3_0_value, alpha_value),
                 linewidth=1.5, label='multi')
    matplot.xlabel(r'$k_1$')
    matplot.ylabel(r'$k_2$')
    matplot.xlim([-0.0, 0.2])
    matplot.ylim([-0.0, 0.2])
    matplot.show()
    return


#отрисовка в фазовой плоскости
def Stream_Plot(k1_value, k1m_value, k2_value, k2m_value, k3_0_value, alpha_value):
    f1 = lambdify((x, y, k1, km1, k3_0, alpha), eq1)
    f2 = lambdify((x, y, k2, km2, k3_0, alpha), eq2)
    Y, X = numpy.mgrid[0:.5:1000j, 0:1:2000j]
    U = f1(X, Y, k1_value, k1m_value, k3_0_value, alpha_value)
    V = f2(X, Y, k2_value, k2m_value, k3_0_value, alpha_value)
    velocity = numpy.sqrt(U*U + V*V)
    matplot.streamplot(X, Y, U, V, density=[2.5, 0.8], color=velocity)
    matplot.xlabel('x')
    matplot.ylabel('y')
    matplot.show()


def Solve_System(init, k1_value, k1m_value, k2_value, k2m_value, k3_0_value, alpha_value, dt, iterations):

    f1 = lambdify((x, y, k1, km1, k3_0,alpha), eq1)
    f2 = lambdify((x, y, k2, km2, k3_0,alpha), eq2)

    def rhs(xy, times):
        return [f1(xy[0], xy[1], k1_value, k1m_value, k3_0_value, alpha_value),
                f2(xy[0], xy[1], k2_value, k2m_value, k3_0_value, alpha_value)]

    times = numpy.arange(iterations) * dt
    return odeint(rhs, init, times), times


#отрисовка колебаний
def Auto_Col():
    solution, times = Solve_System([0.38, 0.22], 0.03, 0.01, 0.05, 0.01, 10, 16, 1e-2, 1e6)

    ax = matplot.subplot(211)
    matplot.plot(times, solution[:, 0])
    matplot.setp(ax.get_xticklabels(), visible=False)
    matplot.ylabel('x')
    matplot.grid()
    matplot.subplot(212, sharex=ax)
    matplot.plot(times, solution[:, 1], color='red')
    matplot.xlabel('t')
    matplot.ylabel('y')
    matplot.grid()
    matplot.show()
    return


#Analysis_K1_K2()
#Stream_Plot(0.03, 0.01, 0.05, 0.01, 10, 16)
Auto_Col()





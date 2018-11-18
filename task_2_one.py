import numpy as numpy
import matplotlib.pyplot as matplot
from matplotlib.legend_handler import HandlerLine2D
from sympy import *

# однопараметрический анализ по параметру k2
# задаем константы
Alpha = 16
k_1 = 0.03
k_m_1 = 0.01
k_m_2 = 0.01
k_3_0 = 10
Tolerance = 1e-15


def Analys_K2(k_1, k_m_1, k_m_2, k_3_0, Alpha, Tolerance):
    # задаем переменные
    y = numpy.linspace(0.001, 0.987, 1000)
    N = numpy.size(y)
    x = numpy.zeros(N)
    z = numpy.zeros(N)
    phi = numpy.zeros(N)
    phi_m = numpy.zeros(N)
    k_2 = numpy.zeros(N)
    sled = numpy.zeros(N)
    Det_A = numpy.zeros(N)

    # заданное значение для pfi
    phi[0] = pow((1 - y[0]), Alpha)
    phi_m[0] = pow((1 - y[0]), Alpha - 1)
    # выражаем x через y для нулевого элемента сетки
    x[0] = k_1 * (1 - y[0]) / (k_1 + k_m_1 + k_3_0 * phi[0] * y[0])
    # выражаем значение k_2 через остальные параметры и переменную y для нулевого элемента сетки
    k_2[0] = (k_m_2 * pow(y[0], 2) * pow((k_1 + k_m_1 + k_3_0 * phi[0] * y[0]), 2) + (k_1 + k_m_1 + k_3_0 * phi[0] * y[
        0]) * k_1 * k_3_0 * phi[0] * y[0] * (1 - y[0])) / (pow((1 - y[0]), 2) * pow((k_m_1 + k_3_0 * phi[0] * y[0]), 2))
    # выпишем элементы матрицы Якоби для нулевого элемента сетки
    a11 = -k_1 - k_m_1 - k_3_0 * phi[0] * y[0]
    a12 = -k_1 - k_3_0 * x[0] + k_3_0 * Alpha * phi_m[0] * y[0] * x[0]
    a21 = -2 * k_2[0] * z[0] - k_3_0 * phi[0] * y[0]
    a22 = -2 * k_2[0] * z[0] - 2 * k_m_2 * y[0] - k_3_0 * phi[0] * x[0] + k_3_0 * Alpha * phi_m[0] * y[0] * x[0]
    # вычисляем след матрицы Якоби
    sled[0] = a11 + a22
    # вычисляем определитель матрицы Якоби
    Det_A[0] = a11 * a22 - a12 * a21

    for i in range(1, N):
        # заданное значение для pfi
        phi[i] = pow((1 - y[i]), Alpha)
        phi_m[i] = pow((1 - y[i]), Alpha - 1)
        # выражаем x через y для различных элементов сетки
        x[i] = k_1 * (1 - y[i]) / (k_1 + k_m_1 + k_3_0 * phi[i] * y[i])
        # выражаем значение k_2 через остальные параметры и переменную y для различных элементов сетки
        k_2[i] = (k_m_2 * pow(y[i], 2) * pow((k_1 + k_m_1 + k_3_0 * phi[i] * y[i]), 2) + (k_1 + k_m_1 + k_3_0 * phi[i]
                                    * y[i]) * k_1 * k_3_0 * phi[i] * y[i] * (1 - y[i])) / (pow((1 - y[i]), 2) * pow((k_m_1 + k_3_0 * phi[i] * y[i]), 2))
        # выпишем элементы матрицы Якоби для различных элементов сетки
        a11 = -k_1 - k_m_1 - k_3_0 * phi[i] * y[i]
        a12 = -k_1 - k_3_0 * phi[i] * x[i] + k_3_0 * Alpha * phi_m[i] * y[i] * x[i]
        a21 = -2 * k_2[i] * z[i] - k_3_0 * phi[i] * y[i]
        a22 = -2 * k_2[i] * z[i] - 2 * k_m_2 * y[i] - k_3_0 * phi[i] * x[i] + k_3_0 * Alpha * phi_m[i] * y[i] * x[i]
        # вычисляем след матрицы Якоби
        sled[i] = a11 + a22
        # вычисляем определитель матрицы Якоби
        Det_A[i] = a11 * a22 - a12 * a21

        # отрисовка графиков и точек бифуркации
        # седло-узловая бифуркация
        if (Det_A[i] * Det_A[i - 1] < Tolerance):
            y_new = y[i - 1] - Det_A[i - 1] * (y[i] - y[i - 1]) / (Det_A[i] - Det_A[i - 1])
            x_new = k_1 * (1 - y_new) / (k_1 + k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new)

            k_2_new = (k_m_2 * pow(y_new,2) * pow((k_1 + k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new), 2) + (k_1 + k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new)
                       * k_1 * k_3_0 * pow((1 - y_new), Alpha) * y_new * (1 - y_new)) / (pow((1 - y_new), 2) * pow((k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new), 2))

            matplot.plot(k_2_new, x_new, 'g*', marker='o')
            matplot.plot(k_2_new, y_new, 'g*', marker='o')

        # бифуркация Хопфа
        if (sled[i] * sled[i - 1] < Tolerance):
            y_new = y[i - 1] - sled[i - 1] * (y[i] - y[i - 1]) / (sled[i] - sled[i - 1])
            x_new = k_1 * (1 - y_new) / (k_1 + k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new)
            k_2_new = (k_m_2 * pow(y_new,2) * pow((k_1 + k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new), 2) + (k_1 + k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new)
                * k_1 * k_3_0 * pow((1 - y_new), Alpha) * y_new * (1 - y_new)) / (pow((1 - y_new), 2) * pow((k_m_1 + k_3_0 * pow((1 - y_new), Alpha) * y_new), 2))

            matplot.plot(k_2_new, x_new, 'k*', marker='o')
            matplot.plot(k_2_new, y_new, 'k*', marker='o')

    matplot.title('Однопараметрический анализ. Завсимость стационарных решений от параметра k_2')
    line1, = matplot.plot(k_2, x, 'b--', label="x")
    line2, = matplot.plot(k_2, y, 'r', label="y")

    matplot.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    matplot.xlim((0, 0.7))
    matplot.xlabel('k_2')
    matplot.ylabel('x,y')
    matplot.grid(True)
    matplot.show()
    return

#задача диапазона констант
alpha_range = [10, 15, 18, 20, 25]
k3_range = [1, 5, 10, 50, 100]

#решение в зависимости от заданных данных
for Element in k3_range:
    #Analys_K2(k_1,k_m_1,k_m_2,k_3_0,Element,Tolerance)
    Analys_K2(k_1, k_m_1, k_m_2, Element, Alpha, Tolerance)

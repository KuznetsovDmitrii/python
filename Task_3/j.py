import matplotlib.pyplot as matplot
from matplotlib.legend_handler import HandlerLine2D

a_mas_verlet = [0.002, 0.05, 0.2]
a_mas_theread = [0.003, 0.2, 1]
a_mas_cython = [0.001, 0.01, 0.15]
x = [10, 100, 200]



for i in range(3):
    a_mas_cython[i] /= a_mas_verlet[i]
    a_mas_cython[i]=1/a_mas_cython[i]
    a_mas_theread[i] /= a_mas_verlet[i]
    a_mas_theread[i]=1/a_mas_theread[i]
    a_mas_verlet[i] /= a_mas_verlet[i]
    a_mas_verlet[i]=1/a_mas_verlet[i]

matplot.title('Ускорение для различного колличества частиц')
line1, = matplot.plot(x, a_mas_verlet, 'b', label="verlet")
line2, = matplot.plot(x, a_mas_cython, 'r', label="verlet_cython")
line3, = matplot.plot(x, a_mas_theread, 'g', label="verlet_threading")

matplot.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

matplot.xlim((0, 200))
matplot.xlabel('N')
matplot.ylabel('Time')
matplot.grid(True)
matplot.savefig('Ускорение1.png')
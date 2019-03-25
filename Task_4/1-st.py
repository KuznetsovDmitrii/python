import numpy
import matplotlib.pyplot as my_plot
import matplotlib.tri as tri
from fenics import *
from mshr import *
import math

R = pi

domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 64)

V = FunctionSpace(mesh, 'P', 1)

bounds = MeshFunction("size_t", mesh, 1)

class boundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] >= 0)

b1 = boundary1()
b1.mark(bounds, 0)

class boundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < 0)

b2 = boundary2()
b2.mark(bounds, 1)

#h = Expression('x[1] * cos(x[0])', degree = 2)
#alpha = 0.5
#R = math.sqrt(pi)
#f = Expression('(1 + alpha) * x[1] * cos(x[0])', degree = 2, alpha = alpha)
#g = Expression('cos(x[0]) * x[1] / R - x[1] * sin(x[0]) * x[0] / R', degree = 2, R = R)

#h = Expression('0.01 * exp(x[0] + x[1])', degree=2)
#alpha = 10
#R = pi
#f = Expression('(alpha) * ( 0.01 * exp(x[0] + x[1])) - 0.01 * exp(x[0]) - 0.01* exp(x[1])', degree=2, alpha=alpha)
#g = Expression('0.01 * exp(x[0]) * x[1]/R + 0.01 * exp(x[1]) * x[0] / R', degree=2, R=R)

h = Expression('0.1 *x[1]*x[0]', degree = 2)
alpha = 0.1
R = pi
f = Expression('0.1 * alpha * x[1] * x[0]', degree = 2, alpha = alpha)
g = Expression('0.1 * x[1]*x[0]/R + 0.1 * x[0]*x[1]/R', degree = 2, R = R)

u = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V, h, bounds, 1)

a = dot(grad(u), grad(v)) * dx + alpha * dot(u, v) * dx
L = f * v * dx - g * v * ds(0, subdomain_data=bounds)

u = Function(V)
solve(a == L, u, bc)

# вычисление ошибок по нормам
error_L2 = errornorm(h, u, 'L2')
print('error_L2 =', error_L2)

h = h.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_C = numpy.max(numpy.abs(vertex_values_u - h))
print('error_C =', error_C)

n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = numpy.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

my_plot.figure()
zfaces = numpy.asarray([h(cell.midpoint()) for cell in cells(mesh)])
my_plot.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
my_plot.title("Аналитическое решение")
my_plot.savefig('1-st_3a1.png')

my_plot.figure()
zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
my_plot.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
my_plot.title("Численное решение")
my_plot.savefig('1-st_3n1.png')

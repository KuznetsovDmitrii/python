import numpy
import matplotlib.pyplot as my_plot
import matplotlib.tri as tri
import math


from fenics import *
from mshr import *

T = 1
steps = 50
dt = T / steps
filenames = []
R = 1
Error = 1E-16
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 64)
V = FunctionSpace(mesh, 'P', 2)

bounds = MeshFunction("size_t", mesh, 1)

# du/dn|y>0 = g(x,y)
class boundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary or (x[1] >= 0)

b1 = boundary1()
b1.mark(bounds, 0)

# u|(y<0) = h(x,y), on boundary and y<0
class boundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary or (x[1] < 0)

b2 = boundary2()
b2.mark(bounds, 1)

alpha = 2
beta = 3
t = 0


#gamma = -1 #main formula
#h = Expression('0.1 * x[0] * cos(x[1])*t', degree = 2, t = t)
#f = Expression('(1 + t) * 0.1 * x[0] * cos(x[1])', degree = 2, t = t)
#g = Expression('-0.1 * x[0] * sin(x[1]) * x[1] * t / R - 0.1 * cos(x[1]) * x[0] * t / R', degree = 2, R = R, t = t)

#gamma = -1 #main formula
#h = Expression('0.5 * x[0] + 0.3 * x[1] + 0.7 * t',degree = 2, t = 0)
#f = Expression('0.7 + 0.5 * x[0] + 0.3 * x[1] + 0.7 * t', degree=2, t = t)
#g = Expression('0.5 * x[0]/R + 0.3 * x[1]/R', degree=2, R = R)


gamma = -10 #main formula
h = Expression('-10 * x[1] * x[0] * x[0] * t', degree = 2, t = t) # == h
f = Expression('-10 * x[1] * x[0] + 20', degree = 2, t = t)
g = Expression('-10 * x[0] * x[1] * x[0] * t / R - 10 * x[0] * x[0] * x[1] * t / R', degree = 2, R = R, t = t)

bc = DirichletBC(V, h, bounds, 1)
u_i = interpolate(h,V)

u = TrialFunction(V)
v = TestFunction(V)

a = gamma*dt * dot(grad(u),grad(v))*dx + u*v*dx
L = (u_i + dt*f)*v*dx + gamma * dt * g * v *ds(0, subdomain_data = bounds)


n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n,d))
triangles = numpy.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:,0],mesh_coordinates[:,1],triangles)

errL2 = []
errC = []
tt = []
u = Function(V)

for n in range(steps):

    t += dt
    tt.append(t)
    h.t = t

    solve(a==L, u, bc)

    u_e = interpolate(h,V)

    vertex_values_u_e = u_e.compute_vertex_values(mesh)
    vertex_values_u   = u.compute_vertex_values(mesh)

    error_L2 = errornorm(u_e, u, 'L2')
    error_C = numpy.max(numpy.abs(vertex_values_u - vertex_values_u_e))

    print('t = ', t, ', error_L2 = ', error_L2,'\n', '        error_C = ', error_C)
    errL2.append(error_L2)
    errC.append(error_C)

    my_plot.figure()
    zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    my_plot.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    #plt.clim(4.1,5.2)
    #plt.clim(-1.5, 1.5)
    my_plot.clim(-0.5, 0.5)
    my_plot.colorbar()
    my_plot.title(('Sol: t = '+"{:.1f}".format(t)+', error_L2 = '+"{:.5f}".format(error_L2)+', error_C = '+ "{:.5f}".format(error_C)))
    my_plot.savefig('heat3_s_'+"{:.1f}".format(t)+'.png')

    my_plot.figure()
    zfaces = numpy.asarray([u_e(cell.midpoint()) for cell in cells(mesh)])
    my_plot.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    my_plot.title(('An: t = '+"{:.1f}".format(t)+', error_L2 = '+"{:.5f}".format(error_L2)+', error_C = '+ "{:.5f}".format(error_C)))
    #plt.clim(4.1,5.2)
    #plt.clim(-1.5, 1.5)
    my_plot.clim(-0.5, 0.5)
    my_plot.colorbar()
    my_plot.savefig('heat3_a_'+"{:.1f}".format(t)+'.png')
    u_i.assign(u)

my_plot.figure()
my_plot.title("Errors")
my_plot.plot(tt, errL2, 'r-', label= 'L2')
my_plot.plot(tt, errC, 'g-', label = 'C')
my_plot.legend()
my_plot.grid()
my_plot.savefig('errors.png')
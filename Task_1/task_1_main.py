import math
import random
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import task_1_interface


class Coordinate:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def Module(self, coordinate):
        return math.sqrt((self.x - coordinate.x) ** 2 + (self.y - coordinate.y) ** 2 + (self.z - coordinate.z) ** 2)


class Speed:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w


class Particle:
    def __init__(self, coordinate, speed, weight, color, time):
        self.coordinate = coordinate
        self.x = self.coordinate.x
        self.y = self.coordinate.y
        self.z = self.coordinate.z
        self.speed = speed
        self.u = self.speed.u
        self.v = self.speed.v
        self.w = self.speed.w
        self.weight = weight
        self.color = color
        self.alive = time


class Interface(QMainWindow, task_1_interface.Ui_MainWindow, QOpenGLWidget):
    part_list = []
    dt = 1000

    def __init__(self):
        super().__init__()
        global dt
        self.Angle_x = 0
        self.Angle_y = 0
        self.zoom = 500
        self.lastPos = QPoint()
        self._color = QColor(255, 0, 0)
        # self.dt = 1000
        dt = 1000
        self.setupUi(self)
        self.timer = QTimer()
        self.timer.timeout.connect(self.draw)
        self.timer.start(dt)
        self.color_button.clicked.connect(self.button_selectColor)
        self.add_button.clicked.connect(self.button_add)
        self.add_random_button.clicked.connect(self.button_add_random)
        self.clear_button.clicked.connect(self.button_clear)
        self.count_combo.currentIndexChanged.connect(self.combobox_numberChoice)
        self.mass.valueChanged.connect(self.slider_valueChanged_mass)
        self.time1.valueChanged.connect(self.slider_valueChanged_time)
        #self.mass.sliderPressed(self.mass1.setText(str(self.mass.sliderPosition())))

    def slider_valueChanged_mass(self):
        #task_1_interface.Ui_MainWindow.mass1.setText(str(self.mass.sliderPosition()))
        self.mass1.setText(str(self.mass.sliderPosition()))

    def slider_valueChanged_time(self):
        # task_1_interface.Ui_MainWindow.mass1.setText(str(self.mass.sliderPosition()))
        self.time.setText(str(self.time1.sliderPosition()))

    def button_clear(self):
        global  part_list
        part_list_new = []
        part_list = part_list_new
        self.gl_sys.update()

    def button_selectColor(self):
        qcolor = QColorDialog.getColor()
        p = self.color_test.palette()
        self._color = QColor(qcolor)
        p.setColor(QPalette.Background, QColor(qcolor))
        self.color_test.setPalette(p)
        self.color_test.show()

    def button_add(self):
        global part_list
        global dt
        emit = Coordinate(float(self.c_x.toPlainText()), float(self.c_y.toPlainText()), float(self.c_z.toPlainText()))
        vel = Speed(float(self.v_x.toPlainText()) / 10000, float(self.v_y.toPlainText()) / 10000,
                       float(self.v_z.toPlainText()) / 10000)
        part_list.append(Particle(emit, vel, float(self.mass.sliderPosition()), self._color.getRgbF(),
                                  dt * int(self.time.toPlainText())))
        print(dt * int(self.time.toPlainText()))
        self.gl_sys.update()

    def button_add_random(self):
        global part_list
        global dt
        part_list.append(
            Particle(Coordinate(random.randint(-500, 500), random.randint(-500, 500), random.randint(-500, 500)),
                     Speed(random.randint(-5, 5) / 10000.0, random.randint(-5, 5) / 10000.0,
                              random.randint(-5, 5) / 10000.0), random.randint(100, 1000),
                     [random.randint(0, 255) / 255.0, random.randint(0, 255) / 255.0, random.randint(0, 255) / 255.0],
                     dt * random.randint(10, 1000)))
        self.gl_sys.update()

    def combobox_numberChoice(self):
        global part_list
        global dt
        self.zoom = 550
        part_count = 0

        if self.count_combo.currentIndex() == 0:
            part_count = 10
        elif self.count_combo.currentIndex() == 1:
            part_count = 35
        elif self.count_combo.currentIndex() == 2:
            part_count = 50
        elif self.count_combo.currentIndex() == 3:
            part_count = 75
        if self.count_combo.currentIndex() != 4:
            dt = 1000
            part_list = []
            part_list.append(Particle(Coordinate(0, 0, 0), Speed(0, 0, 0), 10000, (1, 0, 1), dt * 1000))
            for i in range(1, part_count):
                part_list.append(Particle(
                    Coordinate(random.randint(-500, 500), random.randint(-500, 500), random.randint(-500, 500)),
                    Speed(random.randint(-5, 5) / 2000.0, random.randint(-5, 5) / 2000.0,
                             random.randint(-5, 5) / 2000.0), random.randint(100, 1000),
                    [random.randint(0, 255) / 255.0, random.randint(0, 255) / 255.0, random.randint(0, 255) / 255.0],
                    dt * random.randint(10, 1000)))
        else:
            self.zoom = 775
            dt = 100000
            part_count = 9
            part_list = []
            part_list.append(Particle(Coordinate(0, 0, 0), Speed(0, 0, 0), 332900, [1, 1, 0], dt ** 2))
            part_list.append(
                Particle(Coordinate(0.387, 0, 0), Speed(0, 47870, 0), 0.055, [139 / 255.0, 69 / 255.0, 19 / 255.0],
                         dt ** 2))
            part_list.append(Particle(Coordinate(0.7233, 0, 0), Speed(0, 35020, 0), 0.815,
                                      [255 / 255.0, 160 / 255.0, 122 / 255.0], dt ** 2))
            part_list.append(
                Particle(Coordinate(1, 0, 0), Speed(0, 29760, 0), 1, [30 / 255.0, 144 / 255.0, 255 / 255.0],
                         dt ** 2))
            part_list.append(
                Particle(Coordinate(1.524, 0, 0), Speed(0, 24130, 0), 0.107, [255 / 255.0, 127 / 255.0, 0], dt ** 2))
            part_list.append(
                Particle(Coordinate(3.2, 0, 0), Speed(0, 13070, 0), 318, [233 / 255.0, 150 / 255.0, 122 / 255.0],
                         dt ** 2))
            part_list.append(
                Particle(Coordinate(5.0, 0, 0), Speed(0, 9670, 0), 95.0, [255 / 255.0, 222 / 255.0, 173 / 255.0],
                         dt ** 2))
            part_list.append(
                Particle(Coordinate(10.23, 0, 0), Speed(0, 6840, 0), 14.6, [0, 0, 127 / 255.0], dt ** 2))
            part_list.append(
                Particle(Coordinate(15.07, 0, 0), Speed(0, 5480, 0), 17.1, [30 / 255.0, 100 / 255.0, 255 / 255.0],
                         dt ** 2))
        self.timer.start(dt)
        self.gl_sys.update()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        if event.buttons() & Qt.RightButton:
            self.set_angle_x(0)
            self.set_angle_y(0)
            self.gl_sys.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            self.set_angle_x(self.Angle_x + dy / 100)
            self.set_angle_y(self.Angle_y - dx / 100)

            self.lastPos = event.pos()
            self.gl_sys.update()

    def set_angle_x(self, angle):
        if angle != self.Angle_x:
            self.Angle_x = angle

    def set_angle_y(self, angle):
        if angle != self.Angle_y:
            self.Angle_y = angle

    def wheelEvent(self, event):
        if (self.zoom >= 5) & (self.zoom <= 2500):
            self.zoom -= event.angleDelta().y() / 10
            self.gl_sys.update()
        elif self.zoom < 5:
            self.zoom = 5
        else:
            self.zoom = 2500

    def calculate(self):
        global dt
        G = 6.67408 * (10 ** -11)
        AEM = 149.6 * 10 ** 9
        Earth_mass = 5.974 * 10 ** 24
        dt = 1000
        timerStep = dt
        global part_list
        x_n = []
        y_n = []
        z_n = []
        u_n = []
        v_n = []
        w_n = []
        m_n = []
        col_n = []
        t_n = []
        if self.count_combo.currentIndex() != 4:
            G = 6.67408 * (10 ** -8)
            for partic in part_list:
                for par in part_list:
                    if (partic.coordinate.Module(par.coordinate) > 0) & (
                            partic.coordinate.Module(par.coordinate) < (partic.weight + par.weight) / 100.0):
                        if partic.weight >= par.weight:
                            partic.weight += par.weight
                        else:
                            partic.alive = 0
                if (partic.alive):
                    x_n.append(partic.x)
                    y_n.append(partic.y)
                    z_n.append(partic.z)
                    coordinate = [Coordinate(x, y, z) for x, y, z in zip(x_n, y_n, z_n)]
                    u_n.append(partic.u)
                    v_n.append(partic.v)
                    w_n.append(partic.w)
                    m_n.append(partic.weight)
                    col_n.append(partic.color)
                    t_n.append(int(partic.alive) - timerStep)
        else:
            x_n = [p.x * AEM for p in part_list]
            y_n = [p.y * AEM for p in part_list]
            z_n = [p.z * AEM for p in part_list]
            coordinate = [Coordinate(x, y, z) for x, y, z in zip(x_n, y_n, z_n)]
            u_n = [p.u for p in part_list]
            v_n = [p.v for p in part_list]
            w_n = [p.w for p in part_list]
            m_n = [p.weight * Earth_mass for p in part_list]
            col_n = [p.color for p in part_list]
            timerStep = 100000
            t_n = [p.alive - timerStep for p in part_list]
        length = len(x_n)

        ax_n = []
        ay_n = []
        az_n = []
        for cx, cy, cz in zip(x_n, y_n, z_n):
            part = Coordinate(cx, cy, cz)
            ax = []
            ay = []
            az = []
            for c, m in zip(coordinate, m_n):
                Module = part.Module(c)
                if Module > 0:
                    ax.append(G * m * (c.x - cx) / Module / Module / Module)
                    ay.append(G * m * (c.y - cy) / Module / Module / Module)
                    az.append(G * m * (c.z - cz) / Module / Module / Module)

            ax_n.append(sum(ax))
            ay_n.append(sum(ay))
            az_n.append(sum(az))

        x_n1 = [x + u * timerStep + 0.5 * a * timerStep ** 2
                for x, u, a in zip(x_n, u_n, ax_n)]
        y_n1 = [y + v * timerStep + 0.5 * a * timerStep ** 2
                for y, v, a in zip(y_n, v_n, ay_n)]
        z_n1 = [z + w * timerStep + 0.5 * a * timerStep ** 2
                for z, w, a in zip(z_n, w_n, az_n)]

        ax_n1 = []
        ay_n1 = []
        az_n1 = []
        for cx, cy, cz in zip(x_n1, y_n1, z_n1):
            part = Coordinate(cx, cy, cz)
            ax = []
            ay = []
            az = []
            for x, y, z, m in zip(x_n1, y_n1, z_n1, m_n):
                c = Coordinate(x, y, z)
                Module = part.Module(c)
                if Module > 0:
                    ax.append(G * m * (x - cx) / Module / Module / Module)
                    ay.append(G * m * (y - cy) / Module / Module / Module)
                    az.append(G * m * (z - cz) / Module / Module / Module)

            ax_n1.append(sum(ax))
            ay_n1.append(sum(ay))
            az_n1.append(sum(az))

        u_n1 = [u + 0.5 * (an + an1) * timerStep
                for u, an, an1 in zip(u_n, ax_n, ax_n1)]
        v_n1 = [v + 0.5 * (an + an1) * timerStep
                for v, an, an1 in zip(v_n, ay_n, ay_n1)]
        w_n1 = [w + 0.5 * (an + an1) * timerStep
                for w, an, an1 in zip(w_n, az_n, az_n1)]

        part_list = []
        if self.count_combo.currentIndex() != 4:
            for i in range(length):
                coordinate = Coordinate(x_n1[i], y_n1[i], z_n1[i])
                velocity = Speed(u_n1[i], v_n1[i], w_n1[i])
                part_list.append(Particle(coordinate, velocity, m_n[i], col_n[i], t_n[i]))
        else:
            for i in range(length):
                coordinate = Coordinate(x_n1[i] / AEM, y_n1[i] / AEM, z_n1[i] / AEM)
                velocity = Speed(u_n1[i], v_n1[i], w_n1[i])
                part_list.append(Particle(coordinate, velocity, m_n[i] / Earth_mass, col_n[i], t_n[i]))

        if (self.timer.isActive()):
            self.gl_sys.update()

    def setupGL(self):

        self.gl_sys.initializeGL()
        self.gl_sys.initializeGL = self.initializeGL
        self.gl_sys.paintGL = self.paintGL

    def paintGL(self):
        self.loadScene()
        x_cam = self.zoom * math.sin(self.Angle_y) * math.cos(self.Angle_x)
        y_cam = self.zoom * math.sin(self.Angle_y) * math.sin(self.Angle_x)
        z_cam = self.zoom * math.cos(self.Angle_y)
        gluLookAt(x_cam, y_cam, z_cam, 0, 0, 0, 0, 1, 0)

        if self.timer.isActive():
            self.draw()
            self.calculate()

    def draw(self):
        global part_list
        global dt
        part_list = [p for p in part_list if int(p.alive) > 0]
        if self.count_combo.currentIndex() != 4:
            for i in range(len(part_list)):
                if (abs(part_list[i].x) < 2500) & (abs(part_list[i].y) < 2500) & (abs(part_list[i].z) < 2500):
                    # glMaterialfv(GL_FRONT, GL_AMBIENT, [0.8, 0.8, 0.0])
                    # glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.2, 0.2, 0.2])
                    # glMaterialfv(GL_FRONT, GL_SPECULAR, [0.6, 0.6, 0.6])
                    # glMaterialfv(GL_FRONT, GL_SHININESS, 0.5*128)
                    glPushMatrix()
                    sphere = gluNewQuadric()
                    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, part_list[i].color)
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, part_list[i].color)
                    glTranslatef(part_list[i].x, part_list[i].y, part_list[i].z)
                    gluQuadricDrawStyle(sphere, GLU_FILL)
                    gluSphere(sphere, part_list[i].weight / 100.0, 16, 16)
                    glTranslatef(-part_list[i].x, -part_list[i].y, -part_list[i].z)
                    glPopMatrix()
                    gluDeleteQuadric(sphere)
                else:
                    part_list[i].alive = 0
        else:
            for i in range(len(part_list)):
                if (abs(part_list[i].x) < 2500) & (abs(part_list[i].y) < 2500) & (abs(part_list[i].z) < 2500):
                    glPushMatrix()
                    sphere = gluNewQuadric()
                    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, part_list[i].color)
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, part_list[i].color)
                    glTranslatef(50 * part_list[i].x, 50 * part_list[i].y, 50 * part_list[i].z)
                    gluQuadricDrawStyle(sphere, GLU_FILL)
                    if i == 0:
                        gluSphere(sphere, part_list[i].weight / 24000.0, 16, 16)
                    elif i == 1:
                        gluSphere(sphere, part_list[i].weight * 10, 16, 16)
                    elif i == 2:
                        gluSphere(sphere, part_list[i].weight, 16, 16)
                    elif i == 3:
                        gluSphere(sphere, part_list[i].weight, 16, 16)
                    elif i == 4:
                        gluSphere(sphere, part_list[i].weight * 7, 16, 16)
                    elif i == 5:
                        gluSphere(sphere, part_list[i].weight / 30.0, 16, 16)
                    elif i == 6:
                        gluSphere(sphere, part_list[i].weight / 12.0, 16, 16)
                    elif i == 7:
                        gluSphere(sphere, part_list[i].weight / 2.3, 16, 16)
                    elif i == 8:
                        gluSphere(sphere, part_list[i].weight / 3.2, 16, 16)
                    elif i == 9:
                        gluSphere(sphere, part_list[i].weight * 200, 16, 16)

                    glTranslatef(-50 * part_list[i].x, -50 * part_list[i].y, -50 * part_list[i].z)
                    glPopMatrix()
                    gluDeleteQuadric(sphere)
                else:
                    part_list[i].alive = 0

        label = str(len(part_list))
        self.cur_count.setText(label)

    def initializeGL(self):
        glEnable(GL_CULL_FACE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        global part_list
        part_list = []

    def loadScene(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        x, y, width, height = glGetDoublev(GL_VIEWPORT)
        gluPerspective(90, width / float(height or 1), .25, 5000)
        # glOrtho(-2500., 2500., -2500., 2500., -2500., 2500.)


def main():
    app = QApplication(sys.argv)
    window = Interface()
    window.setupGL()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
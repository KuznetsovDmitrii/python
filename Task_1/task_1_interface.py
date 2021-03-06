# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Desktop\untitled.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import random

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 840)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gl_sys = QtWidgets.QOpenGLWidget(self.centralwidget)
        self.gl_sys.setGeometry(QtCore.QRect(460, 20, 671, 671))
        self.gl_sys.setObjectName("gl_sys")
        self.add_button = QtWidgets.QPushButton(self.centralwidget)
        self.add_button.setGeometry(QtCore.QRect(20, 570, 161, 41))
        self.add_button.setObjectName("add_button")
        self.clear_button = QtWidgets.QPushButton(self.centralwidget)
        self.clear_button.setGeometry(QtCore.QRect(20, 620, 161, 41))
        self.clear_button.setObjectName("clear_button")
        self.add_random_button = QtWidgets.QPushButton(self.centralwidget)
        self.add_random_button.setGeometry(QtCore.QRect(200, 570, 220, 41))
        self.add_random_button.setObjectName("add_random_button")
        self.v_x_label = QtWidgets.QLabel(self.centralwidget)
        self.v_x_label.setGeometry(QtCore.QRect(20, 60, 101, 31))
        self.v_x_label.setObjectName("v_x_label")
        self.v_y_label = QtWidgets.QLabel(self.centralwidget)
        self.v_y_label.setGeometry(QtCore.QRect(20, 100, 101, 31))
        self.v_y_label.setObjectName("v_y_label")
        self.v_z_label = QtWidgets.QLabel(self.centralwidget)
        self.v_z_label.setGeometry(QtCore.QRect(20, 140, 101, 31))
        self.v_z_label.setObjectName("v_z_label")
        self.v_x = QtWidgets.QTextEdit(self.centralwidget)
        self.v_x.setGeometry(QtCore.QRect(110, 63, 75, 31))
        self.v_x.setObjectName("v_x")
        self.v_y = QtWidgets.QTextEdit(self.centralwidget)
        self.v_y.setGeometry(QtCore.QRect(110, 103, 75, 31))
        self.v_y.setObjectName("v_y")
        self.v_z = QtWidgets.QTextEdit(self.centralwidget)
        self.v_z.setGeometry(QtCore.QRect(110, 143, 75, 31))
        self.v_z.setObjectName("v_z")
        self.time_label_new = QtWidgets.QLabel(self.centralwidget)
        self.time_label_new.setGeometry(QtCore.QRect(20, 265, 71, 31))
        self.time_label_new.setObjectName("time_label_new")
        self.time_label = QtWidgets.QLabel(self.centralwidget)
        self.time_label.setGeometry(QtCore.QRect(20, 300, 121, 31))
        self.time_label.setObjectName("time_label")
        self.time = QtWidgets.QTextEdit(self.centralwidget)
        self.time.setGeometry(QtCore.QRect(150, 303, 75, 31))
        self.time.setObjectName("time")
        self.time1 = QtWidgets.QSlider(self.centralwidget)
        self.time1.setGeometry(QtCore.QRect(90, 265, 155, 31))
        self.time1.setOrientation(QtCore.Qt.Horizontal)
        self.time1.setObjectName("mass")
        self.time1.setTracking(True)
        self.time1.setTickInterval(750)
        self.time1.setSingleStep(375)
        self.time1.setRange(100, 10000)
        self.c_x_label = QtWidgets.QLabel(self.centralwidget)
        self.c_x_label.setGeometry(QtCore.QRect(240, 60, 121, 31))
        self.c_x_label.setObjectName("c_x_label")
        self.c_x = QtWidgets.QTextEdit(self.centralwidget)
        self.c_x.setGeometry(QtCore.QRect(350, 63, 75, 31))
        self.c_x.setObjectName("c_x")
        self.c_y_label = QtWidgets.QLabel(self.centralwidget)
        self.c_y_label.setGeometry(QtCore.QRect(240, 100, 121, 31))
        self.c_y_label.setObjectName("c_y_label")
        self.c_y = QtWidgets.QTextEdit(self.centralwidget)
        self.c_y.setGeometry(QtCore.QRect(350, 103, 75, 31))
        self.c_y.setObjectName("c_y")
        self.c_z_label = QtWidgets.QLabel(self.centralwidget)
        self.c_z_label.setGeometry(QtCore.QRect(240, 140, 121, 31))
        self.c_z_label.setObjectName("c_z_label")
        self.c_z = QtWidgets.QTextEdit(self.centralwidget)
        self.c_z.setGeometry(QtCore.QRect(350, 143, 75, 31))
        self.c_z.setObjectName("c_z")
        self.parametrs_part = QtWidgets.QLabel(self.centralwidget)
        self.parametrs_part.setGeometry(QtCore.QRect(120, 10, 320, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.parametrs_part.setFont(font)
        self.parametrs_part.setObjectName("parametrs_part")

        self.color_test = QtWidgets.QWidget(self.centralwidget)
        self.color_test.setGeometry(QtCore.QRect(280, 180, 31, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 207, 139))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 180, 75))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 76, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 102, 7))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 204, 133))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 207, 139))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 180, 75))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 76, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 102, 7))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 204, 133))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 76, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 207, 139))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 180, 75))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 76, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 102, 7))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 76, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 76, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 153, 11))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.color_test.setPalette(palette)
        self.color_test.setAutoFillBackground(True)
        self.color_test.setObjectName("color_test")

        self.mass = QtWidgets.QSlider(self.centralwidget)
        self.mass.setGeometry(QtCore.QRect(80, 183, 155, 31))
        self.mass.setOrientation(QtCore.Qt.Horizontal)
        self.mass.setObjectName("mass")
        self.mass.setTracking(True)
        self.mass.setTickInterval(750)
        self.mass.setSingleStep(375)
        self.mass.setRange(1000, 10000)
        self.mass.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.mass_label = QtWidgets.QLabel(self.centralwidget)
        self.mass_label.setGeometry(QtCore.QRect(20, 180, 71, 31))
        self.mass_label.setObjectName("mass_label")
        self.mass1 = QtWidgets.QTextEdit(self.centralwidget)
        self.mass1.setGeometry((QtCore.QRect(155, 223, 55, 31)))
        self.mass1.setObjectName("mass1")
        self.mass_label_new = QtWidgets.QLabel(self.centralwidget)
        self.mass_label_new.setGeometry(QtCore.QRect(20, 223, 131, 31))
        self.mass_label_new.setObjectName("mass_label_new")
        self.color_label = QtWidgets.QLabel(self.centralwidget)
        self.color_label.setGeometry(QtCore.QRect(240, 180, 71, 31))
        self.color_label.setObjectName("color_label")
        self.method_combo = QtWidgets.QComboBox(self.centralwidget)
        self.method_combo.setGeometry(QtCore.QRect(80, 413, 91, 31))
        self.method_combo.setObjectName("method_combo")
        self.method_combo.addItem("")
        self.alg_label = QtWidgets.QLabel(self.centralwidget)
        self.alg_label.setGeometry(QtCore.QRect(10, 410, 71, 31))
        self.alg_label.setObjectName("alg_label")
        self.count_label = QtWidgets.QLabel(self.centralwidget)
        self.count_label.setGeometry(QtCore.QRect(180, 410, 121, 31))
        self.count_label.setObjectName("count_label")
        self.count_combo = QtWidgets.QComboBox(self.centralwidget)
        self.count_combo.setGeometry(QtCore.QRect(310, 413, 131, 31))
        self.count_combo.setObjectName("count_combo")
        self.count_combo.addItem("")
        self.count_combo.addItem("")
        self.count_combo.addItem("")
        self.count_combo.addItem("")
        self.count_combo.addItem("")
        self.param_sys = QtWidgets.QLabel(self.centralwidget)
        self.param_sys.setGeometry(QtCore.QRect(140, 370, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.param_sys.setFont(font)
        self.param_sys.setObjectName("param_sys")
        self.cur_count_label = QtWidgets.QLabel(self.centralwidget)
        self.cur_count_label.setGeometry(QtCore.QRect(110, 450, 201, 31))
        self.cur_count_label.setObjectName("cur_count_label")
        self.cur_count = QtWidgets.QLabel(self.centralwidget)
        self.cur_count.setGeometry(QtCore.QRect(305, 450, 21, 31))
        self.cur_count.setObjectName("cur_count")
        self.color_button = QtWidgets.QPushButton(self.centralwidget)
        self.color_button.setGeometry(QtCore.QRect(315, 180, 121, 31))
        self.color_button.setObjectName("color_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.c_x.setText(str(random.randint(-500,500)))
        self.c_y.setText(str(random.randint(-500,500)))
        self.c_z.setText(str(random.randint(-500,500)))
        self.time.setText(str(random.randint(100, 10000)))
        self.v_x.setText(str(random.randint(-100,100)))
        self.v_y.setText(str(random.randint(-100,100)))
        self.v_z.setText(str(random.randint(-100,100)))
        self.mass.setValue(random.randint(1000,10000))
        self.mass1.setText(str(self.mass.sliderPosition()))
        self.time1.setValue(random.randint(1000,10000))
        self.time.setText(str(self.time1.sliderPosition()))
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Задание №1"))
        self.add_button.setText(_translate("MainWindow", "Добавить частицу"))
        self.add_random_button.setText(_translate("MainWindow", "Добавить случайную частицу"))
        self.clear_button.setText(_translate("MainWindow", "Очистить сцену"))
        self.v_x_label.setText(_translate("MainWindow", "Скорость X ="))
        self.v_y_label.setText(_translate("MainWindow", "Скорость Y ="))
        self.v_z_label.setText(_translate("MainWindow", "Скорость Z ="))
        self.time_label.setText(_translate("MainWindow", "Значение времени ="))
        self.time_label_new.setText(_translate("MainWindow", "Время"))
        self.c_x_label.setText(_translate("MainWindow", "Координата X ="))
        self.c_y_label.setText(_translate("MainWindow", "Координата Y ="))
        self.c_z_label.setText(_translate("MainWindow", "Координата Z = "))
        self.parametrs_part.setText(_translate("MainWindow", "Задаем параметры частицы:"))
        self.mass_label.setText(_translate("MainWindow", "Масса ="))
        self.mass_label_new.setText(_translate("MainWindow", "Значение массы ="))
        self.color_label.setText(_translate("MainWindow", "Цвет"))
        self.method_combo.setItemText(0, _translate("MainWindow", "Верле"))
        self.alg_label.setText(_translate("MainWindow", "Алгоритм"))
        self.count_label.setText(_translate("MainWindow", "Количество частиц"))
        self.count_combo.setItemText(0, _translate("MainWindow", "10"))
        self.count_combo.setItemText(1, _translate("MainWindow", "35"))
        self.count_combo.setItemText(2, _translate("MainWindow", "50"))
        self.count_combo.setItemText(3, _translate("MainWindow", "75"))
        self.count_combo.setItemText(4, _translate("MainWindow", "Солнечная система"))
        self.param_sys.setText(_translate("MainWindow", "Параметры системы:"))
        self.cur_count_label.setText(_translate("MainWindow", "Текущее количество частиц = "))
        self.cur_count.setText(_translate("MainWindow", "0"))
        self.color_button.setText(_translate("MainWindow", "Выбрать цвет"))
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(902, 546)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 20, 861, 71))
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 30, 581, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        self.OkButton = QtWidgets.QPushButton(self.groupBox)
        self.OkButton.setGeometry(QtCore.QRect(680, 30, 93, 28))
        self.OkButton.setObjectName("OkButton")

        self.Image2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.Image2.setGeometry(QtCore.QRect(340, 110, 551, 381))
        self.Image2.setObjectName("Image2")
        self.Info = QtWidgets.QLabel(self.centralwidget)
        self.Info.setGeometry(QtCore.QRect(40, 90, 191, 16))
        self.Info.setObjectName("Info")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(400, 90, 72, 15))
        self.label_3.setObjectName("label_3")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(40, 110, 281, 381))
        self.plainTextEdit.setObjectName("plainTextEdit")
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 902, 26))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "201900600099 滕翡"))
        self.groupBox.setTitle(_translate("mainWindow", "操作信息"))
        self.comboBox.setItemText(0, _translate("mainWindow", "显示图片"))
        self.comboBox.setItemText(1, _translate("mainWindow", "显示图片信息"))
        self.comboBox.setItemText(2, _translate("mainWindow", "灰度直方图"))
        self.comboBox.setItemText(3, _translate("mainWindow", "直方图均衡化"))
        self.comboBox.setItemText(4, _translate("mainWindow", "灰度化"))
        self.comboBox.setItemText(5, _translate("mainWindow", "平移"))
        self.comboBox.setItemText(6, _translate("mainWindow", "旋转"))
        self.comboBox.setItemText(7, _translate("mainWindow", "翻转"))
        self.comboBox.setItemText(8, _translate("mainWindow", "图像放大（最近邻）"))
        self.comboBox.setItemText(9, _translate("mainWindow", "图像放大（双线性插值）"))
        self.comboBox.setItemText(10, _translate("mainWindow", "图像错切"))
        self.comboBox.setItemText(11, _translate("mainWindow", "灰度图伪彩色"))
        self.comboBox.setItemText(12, _translate("mainWindow", "线性灰度变化"))
        self.comboBox.setItemText(13, _translate("mainWindow", "灰窗级变换"))
        self.comboBox.setItemText(14, _translate("mainWindow", "椒盐噪声"))
        self.comboBox.setItemText(15, _translate("mainWindow", "均值滤波"))
        self.comboBox.setItemText(16, _translate("mainWindow", "高通滤波"))
        self.comboBox.setItemText(17, _translate("mainWindow", "二值图像去噪"))
        self.comboBox.setItemText(18, _translate("mainWindow", "横向锐化"))
        self.comboBox.setItemText(19, _translate("mainWindow", "梯度锐化"))
        self.comboBox.setItemText(20, _translate("mainWindow", "边缘检测算子Sobel"))
        self.comboBox.setItemText(21, _translate("mainWindow", "边缘检测算子Robert"))
        self.comboBox.setItemText(22, _translate("mainWindow", "区域种子"))
        self.comboBox.setItemText(23, _translate("mainWindow", "区域合并分裂"))
        self.OkButton.setText(_translate("mainWindow", "确定"))
        self.Info.setText(_translate("mainWindow", "图片信息与相关说明"))
        self.label_3.setText(_translate("mainWindow", "操作结果"))
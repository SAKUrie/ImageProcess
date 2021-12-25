import sys

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QGraphicsPixmapItem, QGraphicsScene

import MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import BmpFile, BmpReader
import random


class MyApp(MainWindow.Ui_mainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.OkButton.clicked.connect(self.Button)
        self.BMP = BmpFile.ReadBMPFile(BmpReader.path)
        self.random = random.randint(0,2000)

    def Button(self):
        self.message("图片路径已经内置，所有操作基于该图片")
        str = self.comboBox.currentText()
        width = self.BMP.biWidth
        height = self.BMP.biHeight
        if str == "显示图片":
            img = self.BMP.image
            frame = QImage(img, width, height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            # self.item.setScale(self.zoomscale)
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.picshow.setScene(self.scene)  # 将场景添加至视图


    # def message(self, s, title="Info"):
    #     msg = QMessageBox(self)
    #     msg.setWindowTitle(title)
    #     msg.setText(s)
    #     print(s)
    #     msg.show()
    #     msg.buttonClicked.connect(msg.exec_)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

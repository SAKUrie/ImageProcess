# -*- coding: UTF-8 -*-
from struct import unpack
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import math


# 读取并存储 bmp 文件
class ReadBMPFile:
    def __init__(self, filePath):
        file = open(filePath, "rb")
        # 读取 bmp 文件的文件头    14 字节
        self.path = filePath
        self.bfType = unpack("<h", file.read(2))[0]  # 0x4d42 对应BM 表示这是Windows支持的位图格式
        self.bfSize = unpack("<i", file.read(4))[0]  # 位图文件大小
        self.bfReserved1 = unpack("<h", file.read(2))[0]  # 保留字段 必须设为 0
        self.bfReserved2 = unpack("<h", file.read(2))[0]  # 保留字段 必须设为 0
        self.bfOffBits = unpack("<i", file.read(4))[0]  # 偏移量 从文件头到位图数据需偏移多少字节（位图信息头、调色板长度等不是固定的，这时就需要这个参数了）
        # 读取 bmp 文件的位图信息头 40 字节
        self.biSize = unpack("<i", file.read(4))[0]  # 所需要的字节数
        self.biWidth = unpack("<i", file.read(4))[0]  # 图像的宽度 单位 像素
        self.biHeight = unpack("<i", file.read(4))[0]  # 图像的高度 单位 像素
        self.biPlanes = unpack("<h", file.read(2))[0]  # 说明颜色平面数 总设为 1
        self.biBitCount = unpack("<h", file.read(2))[0]  # 说明比特数

        self.biCompression = unpack("<i", file.read(4))[0]  # 图像压缩的数据类型
        self.biSizeImage = unpack("<i", file.read(4))[0]  # 图像大小
        self.biXPelsPerMeter = unpack("<i", file.read(4))[0]  # 水平分辨率
        self.biYPelsPerMeter = unpack("<i", file.read(4))[0]  # 垂直分辨率
        self.biClrUsed = unpack("<i", file.read(4))[0]  # 实际使用的彩色表中的颜色索引数
        self.biClrImportant = unpack("<i", file.read(4))[0]  # 对图像显示有重要影响的颜色索引的数目
        # 像素表
        self.bmp_data = []
        self.Index = np.zeros((self.biHeight, self.biWidth), dtype=np.int)

        if self.biBitCount == 24:
            for height in range(self.biHeight):
                bmp_data_row = []
                # 四字节填充位检测
                count = 0
                for width in range(self.biWidth):
                    bmp_data_row.append(
                        [unpack("<B", file.read(1))[0], unpack("<B", file.read(1))[0], unpack("<B", file.read(1))[0]])
                    count = count + 3
                # bmp 四字节对齐原则
                while count % 4 != 0:
                    file.read(1)
                    count = count + 1
                self.bmp_data.append(bmp_data_row)
            self.bmp_data.reverse()
            file.close()
            # R, G, B 三个通道
            self.R = []
            self.G = []
            self.B = []

            for row in range(self.biHeight):
                R_row = []
                G_row = []
                B_row = []
                for col in range(self.biWidth):
                    B_row.append(self.bmp_data[row][col][0])
                    G_row.append(self.bmp_data[row][col][1])
                    R_row.append(self.bmp_data[row][col][2])
                self.B.append(B_row)
                self.G.append(G_row)
                self.R.append(R_row)

            R = self.R
            G = self.G
            B = self.B
            # 显示图像
            b = np.array(B, dtype=np.uint8)
            g = np.array(G, dtype=np.uint8)
            r = np.array(R, dtype=np.uint8)
            # 显示数组
            self.image = cv2.merge([r, g, b])
        elif self.biBitCount <= 8:
            colornum = 2 ** self.biBitCount
            # 颜色表
            self.color_table = np.zeros((colornum, 3), dtype=np.int)
            for i in range(colornum):
                b = unpack("<B", file.read(1))[0]
                g = unpack("<B", file.read(1))[0]
                r = unpack("<B", file.read(1))[0]
                alpha = unpack("<B", file.read(1))[0]
                self.color_table[i][0] = r
                self.color_table[i][1] = g
                self.color_table[i][2] = b

            width = self.biWidth
            height = self.biHeight
            img = np.zeros((height, width, 3), dtype=np.int)
            self.Index = np.zeros((height, width), dtype=np.int)
            self.image = np.zeros((height, width, 3), dtype=np.int)
            for y in range(height):
                num = 0
                for x in range(width):
                    img_byte = unpack("B", file.read(1))[0]
                    img_byte = bin(img_byte)
                    color_index = breakup_byte(img_byte, self.biBitCount)
                    num += 1
                    for index in color_index:
                        if x < width:
                            img[height - y - 1][x] = self.color_table[index]
                            self.Index[height - y - 1][x] = index
                while num % 4 != 0:  # 每一行的位数都必须为4的倍数
                    num += 1
                    file.read(1)
                num = 0
            self.image = img
        self.gray = np.zeros((self.biHeight, self.biWidth), dtype=np.int)
        if self.biBitCount == 24:
            for i in range(self.biHeight):
                for j in range(self.biWidth):
                    r = self.image[i][j][0]
                    g = self.image[i][j][1]
                    b = self.image[i][j][2]
                    gr = int(0.299 * r + 0.587 * g + 0.114 * b)
                    self.gray[i][j] = gr
        else:
            for i in range(self.biHeight):
                for j in range(self.biWidth):
                    r = self.image[i][j][0]
                    g = self.image[i][j][1]
                    b = self.image[i][j][2]
                    gr = int(0.299 * r + 0.587 * g + 0.114 * b)
                    self.gray[i][j] = gr

    def showImage(self):
        plt.imshow(self.image)
        plt.show()

    def InfoString(self):
        string = ""
        string += "位图文件名称为： %s \n" % (self.path)
        string += "位图文件类型为： %d \n" % (self.bfType)
        string += "位图文件的大小： %d  \n" % (self.bfSize)
        string += "图像宽度： %d 点\n" % (self.biWidth)
        string += "图像高度： %d 点\n" % (self.biHeight)
        string += "图片是"
        b = self.biBitCount
        if b == 0:
            string += "JEPG图"
        elif b == 1:
            string += "单色图"
        elif b == 4:
            string += "16色图"
        elif b == 8:
            string += "256色图"
        elif b == 16:
            string += "64K图"
        elif b == 24:
            string += "16M真彩色图"
        elif b == 32:
            string += "4G真彩色图"
        string += "\n"
        return string

    def Gray(self):
        for x in range(self.biWidth):
            for y in range(self.biHeight):
                r = self.image[y][x][0]
                g = self.image[y][x][1]
                b = self.image[y][x][2]
                grayyy = int(0.299 * r + 0.587 * g + 0.114 * b)
                self.gray[y][x] = grayyy

                self.image[y][x][0] = grayyy
                self.image[y][x][1] = grayyy
                self.image[y][x][2] = grayyy

                if self.biBitCount <= 8:
                    self.color_table[self.Index[y][x]][0] = grayyy
                    self.color_table[self.Index[y][x]][1] = grayyy
                    self.color_table[self.Index[y][x]][2] = grayyy

    def Shift(self, deltax, deltay):
        self.biWidth = max(self.biWidth + deltax, self.biWidth)
        self.biHeight = max(self.biHeight + deltay, self.biHeight)
        height = self.biHeight
        width = self.biWidth

        img = np.zeros((height, width, 3), dtype=np.int)
        index = np.zeros((height, width), dtype=np.int)

        for i in range(height):
            for j in range(width):
                img[i][j] = [0, 0, 0]

        for i in range(height):
            for j in range(width):
                ix = i - deltay
                jx = j - deltax
                if 0 <= i - deltay < height and 0 <= j - deltax < width:
                    img[i][j] = self.image[ix][jx]
                    index[i][j] = self.Index[ix][jx]

        self.image = img
        self.Index = index

    def Flip(self, k=1):
        # 1 水平 0 垂直
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width, 3), dtype=np.int)
        index = np.zeros((height, width), dtype=np.int)

        if k == 0:
            for i in range(height):
                for j in range(width):
                    img[i][j] = self.image[height - i - 1][j]
                    index[i][j] = self.Index[height - i - 1][j]

        if k == 1:
            for i in range(height):
                for j in range(width):
                    img[i][j] = self.image[i][width - j - 1]
                    index[i][j] = self.Index[i][width - j - 1]

        self.image = img
        self.Index = index

    def Rotate(self, theta):
        oldimg = self.image
        h = self.biHeight
        w = self.biWidth
        theta = theta / 180 * math.pi
        self.biWidth = int(abs(w * math.cos(theta) + h * math.sin(theta)))
        self.biHeight = int(abs(w * math.sin(theta) + h * math.cos(theta)))

        rotateX = w // 2
        rotateY = h // 2
        write_rotateX = self.biWidth // 2
        write_rotateY = self.biHeight // 2

        img = np.zeros((self.biHeight, self.biWidth, 3), dtype=np.int)
        index = np.zeros((self.biHeight, self.biWidth), dtype=np.int)

        for i in range(self.biHeight):
            for j in range(self.biWidth):
                img[i][j] = [255, 255, 255]

        for i in range(self.biHeight):
            for j in range(self.biWidth):
                ix = int((j - write_rotateX) * math.sin(theta) + (i - write_rotateY) * math.cos(
                    theta) + rotateY)
                jx = int((j - write_rotateX) * math.cos(theta) - (i - write_rotateY) * math.sin(
                    theta) + rotateX)
                if 0 <= ix < h and 0 <= jx < w:
                    img[i][j] = oldimg[ix][jx]
                    index[i][j] = self.Index[ix][jx]

        self.image = img
        self.Index = index

    def Shrink(self, xtimes, ytimes):
        # self.biWidth = int(self.biWidth*(1-times))
        # self.biHeight = int(self.biHeight*(1-times))
        width = int(self.biWidth * (1 - xtimes))
        height = int(self.biHeight * (1 - ytimes))
        img = np.zeros((self.biHeight, self.biWidth, 3), dtype=np.int)
        index = np.zeros((self.biHeight, self.biWidth), dtype=np.int)

        for i in range(self.biHeight):
            for j in range(self.biWidth):
                img[i][j] = [255, 255, 255]

        for i in range(height):
            for j in range(width):
                ix = int(i / (1 - ytimes))
                jx = int(j / (1 - xtimes))
                img[i][j] = self.image[ix][jx]
                index[i][j] = self.Index[ix][jx]

        self.image = img
        self.Index = index

    def Enlarge_Nearest(self, xtimes, ytimes):
        width = int(self.biWidth * (1 + xtimes))
        height = int(self.biHeight * (1 + ytimes))
        img = np.zeros((height, width, 3), dtype=np.int)
        index = np.zeros((height, width), dtype=np.int)

        for i in range(self.biHeight):
            for j in range(self.biWidth):
                img[i][j] = [0, 0, 0]

        for i in range(height):
            for j in range(width):
                ix = int(i / (1 + ytimes))
                jx = int(j / (1 + xtimes))
                img[i][j] = self.image[ix][jx]
                index[i][j] = self.Index[ix][jx]

        self.image = img
        self.Index = index

    def Enlarge_Interpolation(self, xtimes, ytimes):
        width = int(self.biWidth * (1 + xtimes))
        height = int(self.biHeight * (1 + ytimes))
        img = np.zeros((height, width, 3), dtype=np.int)
        index = np.zeros((height, width), dtype=np.int)

        for i in range(self.biHeight):
            for j in range(self.biWidth):
                img[i][j] = [0, 0, 0]

        for i in range(height):
            for j in range(width):
                #   a,b                a,b+1
                #         nowx,nowy
                #
                #   a+1,b              a+1,b+1
                x = int(i / (1 + ytimes))
                y = int(j / (1 + xtimes))
                xx = i / (1 + ytimes)
                yy = j / (1 + xtimes)
                dx = xx - x
                dy = yy - y
                if x + 1 < self.biHeight and y + 1 < self.biWidth:
                    img[i][j] = dx * dy * self.image[x][y] \
                                + (1 - dx) * dy * self.image[x][y + 1] \
                                + dx * (1 - dy) * self.image[x + 1][y] \
                                + (1 - dx) * (1 - dy) * self.image[x + 1][y + 1]
                else:
                    img[i][j] = self.image[x][y]

        self.image = img
        self.Index = index

    def Shear(self, c, k=1):
        height = self.biHeight
        width = self.biWidth
        if k == 1:
            width = int(self.biWidth + self.biHeight * abs(c))
        else:
            height = int(self.biHeight + self.biWidth * abs(c))
        img = np.zeros((height, width, 3), dtype=np.int)
        index = np.zeros((height, width), dtype=np.int)

        for i in range(height):
            for j in range(width):
                img[i][j] = [255, 255, 255]

        if k == 1:
            for i in range(height):
                for j in range(width):
                    x = i
                    y = int(j - c * i)
                    if 0 <= x < self.biHeight and 0 <= y < self.biWidth:
                        img[i][j] = self.image[x][y]

        elif k == 0:
            for i in range(height):
                for j in range(width):
                    x = int(i - j * c)
                    y = j
                    if 0 <= x < self.biHeight and 0 <= y < self.biWidth:
                        img[i][j] = self.image[x][y]
        self.image = img

    def Hist(self):
        gray_hist = np.zeros(shape=[256])
        self.Gray()
        height = self.biHeight
        width = self.biWidth
        for i in range(height):
            for j in range(width):
                gray_hist[self.gray[i][j]] += 1
        plt.bar(range(len(gray_hist)), gray_hist)  # 画灰度直方图
        plt.show()

    def HistEqualizationGray(self):
        plt.imshow(self.gray, cmap='gray')
        plt.show()
        self.Hist()

        height = self.biHeight
        width = self.biWidth
        prob = np.zeros(shape=256)
        for i in self.gray.ravel():
            prob[i] += 1
        prob = prob / (self.biWidth * self.biHeight)
        prob = np.cumsum(prob)
        img_map = [int(256 * prob[i]) for i in range(256)]

        img = np.zeros((height, width, 3), dtype=np.int)
        for i in range(height):
            for j in range(width):
                img[i, j] = img_map[self.gray[i, j]]

        self.image = img
        plt.imshow(self.image, cmap="gray")
        plt.show()

    def LinearContrast(self, a, b, c, d):
        height = self.biHeight
        width = self.biWidth
        k = (b - a) / (d - c)
        img = np.zeros((height, width, 3), dtype=np.int)
        for i in range(height):
            for j in range(width):
                now = self.gray[i][j]
                if a <= now <= b:
                    img[i][j] = c + k * (now - a)
                else:
                    img[i][j] = now
        self.image = img

    def GrayWindow(self, a, b):
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width, 3), dtype=np.int)
        for i in range(height):
            for j in range(width):
                now = self.gray[i][j]
                if a <= now <= b:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
        self.image = img

    def Colorize(self):
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width, 3), dtype=np.int)
        gray = self.gray
        # for i in range(height):
        #     for j in range(width):
        #         img[i][j][0] = int(max(min(255,(gray[i][j]-127)*(255/55),0)))
        #         img[i][j][1] =int(max(min(255,(gray[i][j]-127)*(255/55),0)))
        #         img[i][j][2] =int(max(min(255,(gray[i][j]-63)*(255/55),0)))

        plt.imshow(gray)
        plt.show()

    def SPNoise(self):
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width, 3), dtype=np.int)
        for i in range(height):
            for j in range(width):
                if np.random.rand(1) > 0.8:
                    img[i][j] = 255
                else:
                    img[i][j] = self.image[i][j]
        self.image = img

    def convolution(self, tmp):
        image = self.gray
        height = self.biHeight
        width = self.biWidth
        x = tmp.shape[0]
        y = tmp.shape[1]
        k = 0
        for i in range(x):
            for j in range(y):
                k += tmp[i][j]
        if k == 0:
            k = 1
        img = np.zeros((height, width), dtype=np.int)
        for i in range(height):
            for j in range(width):
                now = 0
                if i + x < height and j + y < width:
                    for ii in range(x):
                        for jj in range(y):
                            now += image[i + ii][j + jj] * tmp[ii][jj]
                    now /= k
                    img[i + math.ceil(x / 2) - 1][j + math.ceil(y / 2) - 1] = int(now)
                else:
                    img[i][j] = image[i][j]
        self.gray = img
        plt.imshow(self.gray, cmap="gray")
        plt.show()

    def AvgConvolution(self):
        AvgCore = np.zeros(shape=(3, 3), dtype=np.int)
        AvgCore[:][:] = 1
        self.convolution(AvgCore)

    def GaotongConvolution(self):
        Core = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
        self.convolution(Core)

    def BWNoiseElimate(self):
        # 全部认为是四连通
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width, 3), dtype=np.int)
        image = self.image
        for i in range(height):
            for j in range(width):
                if 0 < i < height - 1 and 0 < j < width - 1:
                    around = (int(image[i + 1][j + 1][0]) + int(image[i + 1][j - 1][0]) + int(image[i + 1][j][0]) + int(
                        image[i - 1][j][
                            0]) + int(image[i - 1][j - 1][0]) + int(image[i][j - 1][0]) + int(
                        image[i - 1][j + 1][0]) + int(image[i][j + 1][0])) / 8
                    if abs(around - image[i][j][0]) >= 127.5:
                        img[i][j] = 255 - image[i][j][0]
                    else:
                        img[i][j] = image[i][j][0]
                else:
                    img[i][j] = image[i][j][0]
        self.image = img

    def SharpOne(self):
        # 灰度处理
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width), dtype=np.int)
        gray = self.gray
        for i in range(height - 1):
            for j in range(width - 1):
                img[i][j] = 80 + (gray[i][j + 1] - gray[i][j])
        self.gray = img
        plt.imshow(self.gray, cmap="gray")
        plt.show()

    def SharpTwo(self):
        # 灰度处理
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width), dtype=np.int)
        gray = self.gray
        for i in range(height - 1):
            for j in range(width - 1):
                gd = math.sqrt((gray[i][j + 1] - gray[i][j]) ** 2 + (gray[i + 1][j] - gray[i][j]) ** 2)
                if gd > 20:
                    img[i][j] = min(gd + 130, 255)
                else:
                    img[i][j] = 130
        self.gray = img
        plt.imshow(self.gray, cmap="gray")
        plt.show()

    def Robert(self):
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width), dtype=np.int)
        gray = self.gray
        for i in range(height - 1):
            for j in range(width - 1):
                gd = abs(gray[i + 1][j + 1] - gray[i][j]) ** 2 + abs(gray[i + 1][j] - gray[i][j + 1]) ** 2
                gd = math.sqrt(gd)
                img[i][j] = gd
        self.gray = img
        plt.imshow(self.gray, cmap="gray")
        plt.show()

    def SobelConvolution(self):
        Core1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        Core2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        image = self.gray
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width), dtype=np.int)
        img2 = np.zeros((height, width), dtype=np.int)

        for i in range(height):
            for j in range(width):
                now = 0
                if i + 3 < height and j + 3 < width:
                    for ii in range(3):
                        for jj in range(3):
                            now += image[i + ii][j + jj] * Core1[ii][jj]
                    img[i + math.ceil(3 / 2) - 1][j + math.ceil(3 / 2) - 1] = int(now)
                else:
                    img[i][j] = image[i][j]
        for i in range(height):
            for j in range(width):
                now = 0
                if i + 3 < height and j + 3 < width:
                    for ii in range(3):
                        for jj in range(3):
                            now += image[i + ii][j + jj] * Core2[ii][jj]
                    img2[i + math.ceil(3 / 2) - 1][j + math.ceil(3 / 2) - 1] = int(now)
                else:
                    img2[i][j] = image[i][j]
        for i in range(height):
            for j in range(width):
                img[i][j] = max(img[i][j], img2[i][j])
        self.gray = img
        plt.imshow(self.gray, cmap="gray")
        plt.show()

    def Laplacian(self):
        Core = np.array([[0, -1, 0], [-1, -5, -1], [0, -1, 0]])
        self.convolution(Core)
        Core = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        self.convolution(Core)
        Core = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.convolution(Core)

    def Iterate_Thresh(self, img, initval, MaxIterTimes=20, thre=1):
        """ 阈值迭代算法
        Args:
            img: 灰度图像
            initval: 初始阈值
            MaxIterTimes: 最大迭代次数，默认20
            thre：临界差值，默认为1
        Return:
            计算出的阈值
        """
        mask1, mask2 = (img > initval), (img <= initval)
        T1 = np.sum(mask1 * img) / np.sum(mask1)
        T2 = np.sum(mask2 * img) / np.sum(mask2)
        T = (T1 + T2) / 2
        if abs(T - initval) < thre or MaxIterTimes == 0:
            return T
        return self.Iterate_Thresh(img, T, MaxIterTimes - 1)



    def RegionGerenete(self):
        image = self.gray
        height = self.biHeight
        width = self.biWidth
        img = np.zeros((height, width), dtype=np.int)
        std = image[350,500]
        k = 40  # 设立生长阈值
        isMyArea = 1
        # 开始循环遍历周围像素，种子长大。
        for i in range(height):
            for j in range(width):
                if abs(std-image[i][j])>k:
                    img[i][j]=0
                else:
                    img[i][j]=255
        self.gray = img
        plt.imshow(self.gray, cmap="gray")
        plt.show()

    def GereneteSplit(self):
        image = self.gray
        height = self.biHeight
        width = self.biWidth
        self.function(image, 0, 0, width, height)
        self.gray = image
        plt.imshow(image, cmap='gray')
        plt.show()

    # 判断方框是否需要再次拆分为四个
    def judge(self, img, w0, h0, w, h):
        a = img[h0: h0 + h, w0: w0 + w]
        ave = np.mean(a)
        std = np.std(a, ddof=1)
        count = 0
        total = 0
        for i in range(w0, w0 + w):
            for j in range(h0, h0 + h):
                if abs(img[j, i] - ave) < 1 * std:
                    count += 1
                total += 1
        if (count / total) < 0.96:  # 合适的点还是比较少，接着拆
            return True
        else:
            return False

    ##将图像将根据阈值二值化处理，在此默认125
    def draw(self, img, w0, h0, w, h):
        for i in range(w0, w0 + w):
            for j in range(h0, h0 + h):
                if img[j, i] > 125:
                    img[j, i] = 255
                else:
                    img[j, i] = 0

    def function(self, img, w0, h0, w, h):
        if self.judge(img, w0, h0, w, h) and (min(w, h) > 5):
            self.function(img, w0, h0, int(w / 2), int(h / 2))
            self.function(img, w0 + int(w / 2), h0, int(w / 2), int(h / 2))
            self.function(img, w0, h0 + int(h / 2), int(w / 2), int(h / 2))
            self.function(img, w0 + int(w / 2), h0 + int(h / 2), int(w / 2), int(h / 2))
        else:
            self.draw(img, w0, h0, w, h)

    def saveImage(self, path):
        cv2.imwrite(self.path + ".bmp", self.image)


def byte_to_int(str1):
    # 从一个str类型的byte到int
    result = 0
    for i in range(len(str1)):
        y = int(str1[len(str1) - 1 - i])
        result += y * 2 ** i
    return result


def breakup_byte(num1, n):
    # byte为输入的类型为byte的参数,n为每个数要的位数
    result = []  # 返回的数字
    num = num1[2:]
    num_len = len(num)
    str1 = ""
    for i in range(8 - num_len):
        str1 += str(0)
    num = str1 + num
    for i in range(int(8 / n)):
        temp = num[8 - n * (i + 1):8 - n * i]
        result.append(byte_to_int(temp))
    result.reverse()
    return result


def breakup_16byte(str1, str2):
    # 16位采用小端方式储存
    num1 = str1[2:]
    num2 = str2[2:]
    str1_ = ""
    str2_ = ""
    num_len1 = len(num1)
    num_len2 = len(num2)
    for i in range(8 - num_len1):
        str1_ += str(0)
    num1 = str1_ + num1
    for i in range(8 - num_len2):
        str2_ += str(0)
    num2 = str2_ + num2
    num = num2 + num1
    # 16位用两个字节表示rgb设为555最后一个补0
    result = []
    r = byte_to_int(num[1:6])
    g = byte_to_int(num[6:11])
    b = byte_to_int(num[11:16])
    result.append(r * 8)
    result.append(g * 8)
    result.append(b * 8)
    return result

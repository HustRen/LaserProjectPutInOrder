# coding:utf-8
import abc
import math
import os
import sys
import random

import cv2
import numpy as np

from tools import file_name, mkdir, traversalDir_FirstDir, GetSatrtXYFromPolar

class SimAlgorSuper(metaclass = abc.ABCMeta):
    def __init__(self, im_size):
        self._data = np.zeros(im_size)

    @abc.abstractclassmethod
    def data(self):
        pass

class Noise(SimAlgorSuper):
    def __init__(self, im_size, laser):
        super().__init__(im_size)
        self.__laser = laser
    
    def data(self):
        imgRow = self._data.shape[0]
        imgCol = self._data.shape[1]
        for row in range(0, imgRow):
            for col in range(0, imgCol):
                gay =  int(self.__laser[row][col])
                self._data[row][col] = min(gay, int(255))
        return self._data

class LaserSpot(SimAlgorSuper):
    def __init__(self, im_size, position, radius):
        super().__init__(im_size)
        self.__spotRow = position[0] if position[0] >= 0 else 0
        self.__spotCol = position[1] if position[1] >= 0 else 0
        self.__mainSpotRadius = radius
        self.__secSpotPara_A = 267.8
        self.__secSpotPara_a = -0.707
        self.__secSpotPara_b = -0.7621
    
    def data(self):
        rowUp = min(self.__spotRow + self.__mainSpotRadius + 1, self._data.shape[0])
        rowDown = max(self.__spotRow - self.__mainSpotRadius, 0)
        colUp = min(self.__spotCol + self.__mainSpotRadius + 1, self._data.shape[1])
        colDown = max(0, self.__spotCol - self.__mainSpotRadius)
        for row in range(rowDown, rowUp):
            for col in range(colDown, colUp):
                d = np.sqrt(pow(row - self.__spotRow, 2) + pow(col - self.__spotCol, 2))
                if d - self.__mainSpotRadius < 0.51:
                    self._data[row][col] = 255
        for row in range(0, self._data.shape[0]):
            for col in range(0, self._data.shape[1]):
                d = np.sqrt(pow(row - self.__spotRow, 2) + pow(col - self.__spotCol, 2))
                if d > self.__mainSpotRadius:
                    index = 2 / (1 + np.exp(pow(self.__mainSpotRadius, -0.7621) * (d - self.__mainSpotRadius)))
                    self._data[row][col] = int(255 * index)
        return self._data
    
class Target(SimAlgorSuper):
    def __init__(self, im_size, target, r, angle):
        super().__init__(im_size)
        self.__target = target
        self.__r = r
        self.__angle = angle
    
    def data(self):
        imgRow, imgCol= self._data.shape
        targetRow, targetCol = self.__target.shape

        startRow, startCol = GetSatrtXYFromPolar(self._data.shape, self.__target.shape, self.__r, self.__angle)
        for row in range(0, targetRow):
            for col in range(0, targetCol):
                gay = int(self.__target[row][col])
                self._data[startRow + row][startCol + col] = min(gay, int(255))
        return self._data

class TeargetPixel(SimAlgorSuper):
    def __init__(self, im_size, target, centorRow, centorCol):
        super().__init__(im_size)
        self.__target = target
        self.__centorRow = centorRow
        self.__centorCol = centorCol
    
    def data(self):
        imgRow, imgCol= self._data.shape
        targetRow, targetCol = self.__target.shape

        # startRow = 0 if self.__centorRow - int(targetRow / 2) < 0 else self.__centorRow - int(targetRow / 2)
        # startRow = startRow if startRow < imgRow - targetRow else imgRow - targetRow - 1
        # startCol = 0 if self.__centorCol - int(targetCol / 2) < 0 else self.__centorCol - int(targetCol / 2)
        # startCol = startCol if startCol < imgCol - targetCol else imgCol - targetCol - 1
        startRow = self.__centorRow - 53
        startCol = self.__centorCol - 49
        for row in range(0, targetRow):
            for col in range(0, targetCol):
                gay = int(self.__target[row][col])
                self._data[startRow + row][startCol + col] = min(gay, int(255))
        return self._data


class Background(SimAlgorSuper):
    def __init__(self, im_size, bk_image):
        super().__init__(im_size)
        self.__bk = bk_image

    def data(self):
        self._data =self.__bk
        return self._data

class SimImage():
    def __init__(self, im_size):
        self.__image = np.zeros(im_size)
        self.__datas = []
    
    def add(self, sim):
        self.__datas.append(sim.data())

    def image(self):
        for data in self.__datas:
            self.__image = self.__image + data
            self.__image[self.__image > 255] = 255
        return self.__image.astype(np.uint8)    
                
def main():
    scrPath  = 'D:/LaserData/background/1024X1024/scr/airplane/airplane_002.jpg'
    scr    = cv2.imread(scrPath, cv2.IMREAD_GRAYSCALE)
    im = SimImage(scr.shape)
    im.add(Background(scr.shape, scr))
    #im.add(Noise(scr.shape, laser))
    im.add(LaserSpot(scr.shape, (100, 200), 10))
    image = im.image()
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sim():
    base = 'D:/LaserData/background/1024X1024/batch2/'
    filelist = file_name('D:/LaserData/background/1024X1024/scr/airplane', '.jpg', True)
    for file in filelist:
        tname = os.path.basename(file) 
        (shotname,extension) = os.path.splitext(tname)#文件名、后缀名
        floder = base + shotname
        mkdir(floder)
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        row, col = im.shape
        R = [10, 20, 30, 40, 50, 60]
        for r in R:
            for i in range(4):
                simImg = SimImage(im.shape)
                simImg.add(Background(im.shape, im))
                r0 = random.randint(60, row - 60)
                c0 = random.randint(60, col - 60)
                simImg.add(LaserSpot(im.shape, (r0, c0), r))
                name = str(r) + '_' + str(r0) + '_' + str(c0) + '.png'
                cv2.imwrite(floder + '/' + name, simImg.image())
                print(name)


if __name__ == "__main__":
    file = 'D:/LaserData/ans/template/template_0_0.png'
    tname = os.path.basename(file) 
    (shotname,extension) = os.path.splitext(tname)#文件名、后缀名
    floder = 'D:/LaserData/ans/' + shotname
    mkdir(floder)
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    row, col = im.shape
    R = [10, 20, 30, 40, 50, 60]
    for r in R:
        for i in range(4):
            simImg = SimImage(im.shape)
            simImg.add(Background(im.shape, im))
            r0 = random.randint(60, row - 60)
            c0 = random.randint(60, col - 60)
            simImg.add(LaserSpot(im.shape, (r0, c0), r))
            name = str(r) + '_' + str(r0) + '_' + str(c0) + '.png'
            cv2.imwrite(floder + '/' + name, simImg.image())
            print(name)
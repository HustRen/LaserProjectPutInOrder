# coding:utf-8
import os
import sys

import time

import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

from textureSynthesis import OcclusionEstimateImage
from textureSynthesis import GridImage

sys.path.append(os.path.dirname(__file__) + os.sep + '../../lib')
from imageQualityAlgor import MFSIM,SNR,SSIM

def detect_circles_demo(image):
    dst = cv2.pyrMeanShiftFiltering(image, 10, 100)   #边缘保留滤波EPF
    cv2.imshow('test', dst)
    cimage = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 20, param1=60, param2=20, minRadius=5, maxRadius=100)
    if type(circles) != None:
        for i in circles[0, : ]:
            i = np.uint16(np.around(i)) #把circles包含的圆心和半径的值变成整数
            #if image[i[0]][i[1]][0] > 200:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  #画圆
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)  #画圆心
    cv2.imshow("circles", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

class SpotEstimate():
    def __init__(self, scrImage):
        if scrImage.ndim == 3:
            self.occlusionImage = cv2.cvtColor(scrImage, cv2.COLOR_RGB2GRAY)
        else:
            self.occlusionImage = scrImage
        self.mask = self.getMask(self.occlusionImage)

    def getEstimateImg(self):
        eImage = OcclusionEstimateImage(self.occlusionImage, self.mask, 15)
        eImage.debugModel('D:/LaserData//plane/estimate/secondySpot')
        return eImage.textureSynthesis()

    @staticmethod
    def getMask(occlusionImage, dtype = 'circles'):
        if dtype == 'circles':
            image = cv2.cvtColor(occlusionImage, cv2.COLOR_GRAY2RGB)
            dst = cv2.pyrMeanShiftFiltering(image, 10, 100)
            cimage = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
            circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0) 
            circle = circles[0, 0, :]
            mask3 = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask3, (circle[0], circle[1]), circle[2], (255,255,255), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask3, cv2.COLOR_RGB2GRAY)
        else:
            rStr, rowStr, colStr=dtype.split('_')
            image = cv2.cvtColor(occlusionImage, cv2.COLOR_GRAY2RGB)
            mask3 = np.zeros(image.shape, dtype=np.uint8)
            r = int(rStr)
            if r < 60:
                R = r + int((0.8 - 0.01*r)*r)
            else:
                R = r + int(0.3*r)
            cv2.circle(mask3, (int(colStr), int(rowStr)), R, (255,255,255), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask3, cv2.COLOR_RGB2GRAY)
        return mask

class VarMeanEstimate(object):
    def __init__(self, image, grid, dtype = 'circles'):
        mask = SpotEstimate.getMask(image, dtype)
        self.gridImage = GridImage(image, grid)
        size = self.gridImage.meanImage.shape
        scaleMask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        menEst = OcclusionEstimateImage(self.gridImage.meanImage, scaleMask, 3)
        self.meanImg = menEst.textureSynthesis()
        varEst = OcclusionEstimateImage(self.gridImage.varImage, scaleMask, 3)
        self.varImg = varEst.textureSynthesis()

    def show(self):
        plt.subplot(2,2,1)
        plt.title('scrmean')
        plt.imshow(self.gridImage.meanImage)

        plt.subplot(2,2,2)
        plt.title('scrVar')
        plt.imshow(self.gridImage.varImage)

        plt.subplot(2,2,3)
        plt.title('estMean')
        plt.imshow(self.meanImg)

        plt.subplot(2,2,4)
        plt.title('estVar')
        plt.imshow(self.varImg)

        plt.show()
        plt.waitforbuttonpress()

def getfilename(fullpath):
    tname = os.path.basename(fullpath) 
    (shotname,extension) = os.path.splitext(tname)#文件名、后缀名
    return shotname

def MFSIMforMatlab(im1Path, im2Path):
    im1 = cv2.imread(im1Path, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(im2Path, cv2.IMREAD_GRAYSCALE)
    score = MFSIM(im1, im2)
    return score * 100

def getEstimateFeature(fullfilepath):
    image = cv2.imread(fullfilepath, cv2.IMREAD_GRAYSCALE)
    name = getfilename(fullfilepath)
    girdEst = GridImage(image, 3)
    match = re.search(r'[0-9]{1,2}_[0-9]{1,3}_[0-9]{1,3}', name)
    if match:
        vm = VarMeanEstimate(image, 3, name)
        f1 = SNR(girdEst.meanImage)
        f2 = SNR(vm.meanImg)
        f3 = SSIM(vm.meanImg, girdEst.meanImage)
        f4 = SSIM(vm.varImg, girdEst.varImage)
        rStr, rowStr, colStr = name.split('_')
        r = int(rStr)
        f5 = float(r * r * 3.1415) / (image.shape[0] * image.shape[1])
    else:
        f1 = 0.0
        f2 = 0.0
        f3 = 0.0
        f4 = 0.0
        f5 = 0.0
    #image[image < 255] = 0
    #f5 = np.sum(np.sum(image, axis=1),axis=0) / (image.shape[0] * image.shape[1] * 255)  
    return [f1, f2, f3, f4, f5]

def getScore(fullfilepath):
    path = 'D:/LaserData/background/1024X1024/scr/airplane/'
    name = fullfilepath.split('/')
    rePath = path + name[-2] + '.jpg'
    im1 = cv2.imread(fullfilepath, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(rePath, cv2.IMREAD_GRAYSCALE)
    ssim = WMSSIM.SSIM(im1,im2)
    mfsim = WFSIM.MFSIM(im1,im2)
    return [ssim, mfsim]

def testSample():
    floder = 'D:/LaserData/background/1024X1024/batch2/airplane_298/'
    filename = '10_163_87' # 10_163_87 0.7 // 20_115_109 0.6 // 30_66_109 0.5 // 40_62_169 0.4 // 50_85_63 0.3
    image = cv2.imread(floder+'/'+filename + '.png', cv2.IMREAD_GRAYSCALE)
    showImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rStr, rowStr, colStr = filename.split('_')
    r = int(rStr)
    if r < 60:
        R = r + int((0.8 - 0.01*r)*r)
    else:
        R = r + int(0.3*r)
    cv2.circle(showImage, (int(colStr), int(rowStr)), R, (255,0,0))
    mask3 = np.zeros(showImage.shape, dtype=np.uint8)
    cv2.circle(mask3, (int(colStr), int(rowStr)), R, (255,255,255), thickness=cv2.FILLED)
    mask = cv2.cvtColor(mask3, cv2.COLOR_RGB2GRAY)

    # test = OcclusionEstimateImage(image, mask, 15, attenuation = 80, truncation = 0.8)
    # test.debugModel('D:/LaserData/plane/estimate/newmodel_paper')
    # esImage = test.textureSynthesis()
    cv2.imshow('sample', showImage)
    cv2.imshow('mask', mask)
    # cv2.imshow('esImage', esImage)
    # cv2.imwrite('D:/LaserData/mask.png', mask)
    # cv2.imwrite('D:/LaserData/circledection.png', showImage)
    cv2.waitKey()
    cv2.destroyAllWindows()

def compute_ent():
    laser = cv2.imread('D:/LaserData/background/1024X1024/batch2/airplane_298/10_87_90.png', cv2.IMREAD_GRAYSCALE)
    nolaser = cv2.imread('D:/LaserData/background/1024X1024/scr/airplane/airplane_298.jpg', cv2.IMREAD_GRAYSCALE)
    est = cv2.imread('D:/LaserData/background/est.png', cv2.IMREAD_GRAYSCALE)
    row = 87
    col = 90
    r = 15
    aim = nolaser[row - r:row + r, col - r: col + r]
    bim = laser[row - r:row + r, col - r: col + r]
    cim = est[row - r:row + r, col - r: col + r]

    cv2.imshow('nolaser', aim)
    cv2.imshow('laser', bim)
    cv2.imshow('est', cim)
    cv2.imshow('estScr', est)
    cv2.waitKey()
    cv2.destroyAllWindows()
    a = calc_ent(aim)
    b = calc_ent(bim)
    c = calc_ent(cim)

    print('nolaser:%f laser:%f est:%f'%(a,b,c))


def calc_ent(xmat):
    """
        calculate shanno ent of x
    """
    
    x = xmat.flatten()
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def main():
    scrImage = cv2.imread('D:/LaserData/background/1024X1024/scr/airplane/airplane_298.jpg', cv2.IMREAD_GRAYSCALE)
    maskImage = cv2.imread('D:/LaserData/mask.png', cv2.IMREAD_GRAYSCALE)
    disImage = cv2.imread('D:/LaserData/background/1024X1024/batch2/airplane_298/10_87_90.png', cv2.IMREAD_GRAYSCALE)
    estImage = cv2.imread('D:/LaserData/plane/estimate/newmodel_paper/1009.png', cv2.IMREAD_GRAYSCALE)
    print('-------------------------scrimage-----------------------------')
    print('scrImage SNR: %f', (WFSIM.SNR(scrImage)))
    print('disImage SNR: %f', (WFSIM.SNR(disImage)))
    print('estImage SNR: %f', (WFSIM.SNR(estImage)))
    print('scrImage and disImage MFSIM: %f', (WFSIM.MFSIM(scrImage, disImage)))
    print('estImage and disImage MFSIM: %f', (WFSIM.WFSIM(estImage, disImage)))
    print('scrImage and disImage WFSIM: %f', (WFSIM.WFSIM(scrImage, disImage)))
    print('estImage and disImage WFSIM: %f', (WFSIM.WFSIM(estImage, disImage)))
    print('scrImage and disImage WMS_SSIM: %f', (WMSSIM.WMS_SSIM(scrImage, disImage)))
    print('estImage and disImage WMS_SSIM: %f', (WMSSIM.WMS_SSIM(estImage, disImage)))
    print('scrImage and disImage SSIM: %f', (WMSSIM.SSIM(scrImage, disImage)))
    print('estImage and disImage SSIM: %f', (WMSSIM.SSIM(estImage, disImage)))
    print('-------------------------gridimage mean----------------------------')
    gridScr = GridImage(scrImage, 3)
    gridDist = GridImage(disImage, 3)
    vm = VarMeanEstimate(disImage, 3, dtype='10_87_90')
    vm.show()
    print('Scr mean SNR: %f', (WFSIM.SNR(gridScr.meanImage)))
    print('dist mean SNR: %f', (WFSIM.SNR(gridDist.meanImage)))
    print('gird mean SNR: %f', (WFSIM.SNR(vm.meanImg)))
    print('gird scrImage and disImage SSIM: %f', (WMSSIM.SSIM(gridScr.meanImage, gridDist.meanImage)))
    print('gird estImage and disImage SSIM: %f', (WMSSIM.SSIM(vm.meanImg, gridDist.meanImage)))
    print('gird scrImage and disImage lumComp: %f, conComp: %f', (WFSIM.lumComp(gridScr.meanImage, gridDist.meanImage), WFSIM.conComp(gridScr.meanImage, gridDist.meanImage)))
    print('gird estImage and disImage lumComp: %f, conComp: %f', (WFSIM.lumComp(vm.meanImg, gridDist.meanImage), WFSIM.conComp(vm.meanImg, gridDist.meanImage)))
    print('-------------------------gridimage var-----------------------------')
    print('gridScr SNR: %f', (WFSIM.SNR(gridScr.varImage)))
    print('gridDist SNR: %f', (WFSIM.SNR(gridDist.varImage)))
    print('gird SNR: %f', (WFSIM.SNR(vm.varImg)))
    print('gird scrImage and disImage SSIM: %f', (WMSSIM.SSIM(gridScr.varImage, gridDist.varImage)))
    print('gird estImage and disImage SSIM: %f', (WMSSIM.SSIM(vm.varImg, gridDist.varImage)))

if __name__ == "__main__":
    compute_ent()

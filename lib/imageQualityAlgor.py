# coding:utf-8
import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image
from scipy.ndimage import filters

def PSNR(im1,im2):
    '''[峰值信噪比]
    
    Arguments:
        im1 {[numpy-array-2d]} -- [参考图/待评价图像]
        im2 {[numpy-array-2d]} -- [参考图/待评价图像]
    
    Returns:
        [double] -- [返回两幅图像的峰值信噪比]
    '''

    h,w = im1.shape
    diff = im1.astype(np.float) - im2.astype(np.float)
    diff = diff * diff
    rmse = diff.sum() / (h * w)
    psnr = 20*np.log10(255/(np.sqrt(rmse) + 0.001))
    return psnr

def SNR(im):
    '''[信噪比：均值/方差]
    
    Arguments:
        im {[numpy-array-2d]} -- [二维数组，表示待评价图像]
    
    Returns:
        [double] -- [图像的信噪比]
    '''

    imfloat = im.astype(np.float)
    mu = imfloat.mean()
    sigma = imfloat.std()
    return 10 * np.log10(mu / sigma)

def MSE(im1, im2):
    '''[量幅图的均方误差]
    
    Arguments:
        im1 {[numpy-ayyay-2d]} -- [参考图/待评价图像]
        im2 {[numpy-ayyay-2d]} -- [参考图/待评价图像]
    
    Returns:
        [double] -- [两幅图的均方误差]
    '''

    h,w = im1.shape
    diff = im1.astype(np.float) - im2.astype(np.float)
    diff = diff * diff
    mse = diff.sum() / (h * w)
    return mse

def MSESNR(im1, im2):
    '''[基于均方误差的信噪比]
    
    Arguments:
        im1 {[numpy-array-2d]} -- [参考图/待评价图像]
        im2 {[numpy-array-2d]} -- [参考图/待评价图像]
    
    Returns:
        [double] -- [返回两幅图像的信噪比]
    '''
    mse = MSE(im1, im2)
    snr = -10 * np.log10(mse)
    return snr

def SSIM(im1, im2, k = [0.01, 0.03]):
    '''[结构相似度]
    
    Arguments:
        im1 {[numpy-array-2d]} -- [参考图/待评价图像]
        im2 {[numpy-array-2d]} -- [参考图/待评价图像]
    
    Returns:
        [double] -- [返回两幅图像的结构相似度]
    '''
    if im1.shape != im2.shape:
        print("shape error!")
        return
    scrImg = im1.astype(np.float)
    laserImg = im2.astype(np.float)

    c1 = (k[0]*255)**2
    c2 = (k[1]*255)**2
    win = gaussian(11, 1.5)

    mu1 = filters.correlate(scrImg, win)
    mu1_sq = mu1*mu1  #对应元素乘积
    s1sq = filters.correlate(scrImg*scrImg, win) - mu1_sq
    mu2 = filters.correlate(laserImg, win)
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    s2sq = filters.correlate(laserImg*laserImg, win)-mu2_sq
    s12 = filters.correlate(scrImg*laserImg, win)-mu1_mu2

    ssims = ((2*mu1_mu2 + c1)*(2*s12 + c2)) / ((mu1_sq + mu2_sq + c1)*(s1sq + s2sq + c2))
    return ssims.mean()

'''
基于光斑特征的激光干扰图像质量评估算法WFSIM--见中科院钱方的论文
WFSIM算法包括MFSIM算法和加权因子：
    A.MFSIM算法,MFSIM算法包含4个比较函数：
        1.lumComp:        亮度对比函数
        2.conComp:        对比度比较函数
        3.edgeComp:       边缘清晰度函数
        4.locFeatureComp  局部特征点保持度函数
    B.加权因子：
        1.SNRxy:          信噪比加权因子
        2.satPixelRate    饱和像素率加权因子 
'''

def lumComp(im1, im2):
    if im1.shape != im2.shape:
        print("shape error!")
        return
    Lx = np.log10(im1.astype(np.float) + 0.001)
    Ly = np.log10(im2.astype(np.float) + 0.001)
    LxSeq = Lx * Lx
    LySeq = Ly * Ly
    Lssim = (2 * Lx * Ly + 0.001) / (LxSeq + LySeq + 0.001)
    return Lssim.mean()

def conComp(im1, im2):
    if im1.shape != im2.shape:
        print("shape error!")
        return
    x = im1.astype(np.float)
    y = im2.astype(np.float)
    ux = x.mean()
    uy = y.mean()
    Cx = x / ux
    Cy = y / uy
    CxSeq = Cx * Cx
    CySeq = Cy * Cy
    Cssim = (2 * Cx * Cy + 0.001) / (CxSeq + CySeq + 0.001)
    return Cssim.mean()

def edgeComp(im1, im2):
    if im1.shape != im2.shape:
        print("shape error!")
        return
    sobelx = cv2.Sobel(im1.astype(np.float), cv2.CV_64F, 1, 0)  # x方向的梯度
    sobely = cv2.Sobel(im1.astype(np.float), cv2.CV_64F, 0, 1)  # y方向的梯度
    Gx = np.absolute(sobelx) + np.absolute(sobely)
    GxSeq = Gx * Gx

    sobelx = cv2.Sobel(im2.astype(np.float), cv2.CV_64F, 1, 0)  # x方向的梯度
    sobely = cv2.Sobel(im2.astype(np.float), cv2.CV_64F, 0, 1)  # y方向的梯度
    Gy = np.absolute(sobelx) + np.absolute(sobely)
    GySeq = Gy * Gy
    Dssim = (2 * Gx * Gy + 0.001) / (GxSeq + GySeq + 0.001)
    return Dssim.mean()

def locFeatureComp(im1, im2):
    if im1.shape != im2.shape:
        print("shape error!")
        return
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True, type=cv2.FastFeatureDetector_TYPE_9_16)
    fx = fast.detect(im1, None)
    fy = fast.detect(im2, None)
    a = 0
    for xkey in fx:
        for ykey in fy:
            if xkey.pt == ykey.pt:
                a = a + 1
    b = len(fx) - a
    c = len(fy) - a
    ans = float(a) / float(a + b + c)
    return ans

def MFSIM(im1, im2):
    l = lumComp(im1, im2)
    c = conComp(im1, im2)
    d = edgeComp(im1, im2)
    r = locFeatureComp(im1, im2)
    return l * c * d * r

def satPixelRate(imlaser):
    Ns = 0
    for e in imlaser.flat:
        if e > 245:
            Ns = Ns + 1
    row, col = imlaser.shape
    return Ns / (row * col)

def SNRxy(imScr, imLaser):
    return SNR(imLaser) / SNR(imScr)

def WFSIM(imScr, imLasr):
    N = satPixelRate(imLasr)
    snr = SNRxy(imScr, imLasr)
    mfsim = MFSIM(imScr, imLasr)
    return (1 - N) * snr * mfsim

'''
基于小波加权的激光干扰效果评估WMS_SSIM--见中科院钱方的论文
'''
def gaussian(size, sigma):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    xc = (size - 1) / 2
    yc = (size - 1) / 2

    gauss = np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def lumSimi(im1, im2):
    if im1.shape != im2.shape:
        print("shape error!")
        return
    scrImg = im1.astype(np.float)
    laserImg = im2.astype(np.float)

    c1 = (0.01*255)**2
    win = gaussian(11, 1.5)

    mu1 = filters.correlate(scrImg, win)
    mu1_sq = mu1*mu1  #对应元素乘积
    mu2 = filters.correlate(laserImg, win)
    mu2_sq = mu2*mu2
    lum = (2 * mu1 * mu2 + c1) / (mu1_sq + mu2_sq + c1)
    return lum.mean()

def conStuSimi(im1, im2):
    if im1.shape != im2.shape:
        print("shape error!")
        return
    scrImg = im1.astype(np.float)
    laserImg = im2.astype(np.float)
    c2 = (0.03*255)**2
    win = gaussian(11, 1.5)

    mu1 = filters.correlate(scrImg, win)
    mu1_sq = mu1 * mu1  # 对应元素乘积
    s1sq = filters.correlate(scrImg * scrImg, win) - mu1_sq
    mu2 = filters.correlate(laserImg, win)
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    s2sq = filters.correlate(laserImg * laserImg, win) - mu2_sq
    s12 = filters.correlate(scrImg * laserImg, win) - mu1_mu2

    csmi = (2*s12 + c2) / (s1sq + s2sq + c2)
    return csmi.mean()

def WMS_SSIM(im1, im2):
    waveListScr = pywt.wavedec2(data=im1, wavelet='sym8', level=4)
    waveListLas = pywt.wavedec2(data=im2, wavelet='sym8', level=4)

    scrLL4 = waveListScr[0]
    lasLL4 = waveListLas[0]
    ssim4 = lumSimi(scrLL4, lasLL4)
    ssimj = 0.0
    whl = np.array([0.6024, 2.0456, 2.6682, 2.1512])
    whh = np.array([0.1460, 1.0285, 2.0004, 1.9643])
    for i in range(1, 5):  #for(int i = 1; i < 5; i++)
        ssimj = ssimj + whl[i - 1] * conStuSimi(waveListScr[i][0], waveListLas[i][0])
        ssimj = ssimj + whl[i - 1] * conStuSimi(waveListScr[i][1], waveListLas[i][1])
        ssimj = ssimj + whh[i - 1] * conStuSimi(waveListScr[i][2], waveListLas[i][2])

    total = (ssim4 + ssimj) / (2 * whl.sum() + whh.sum() + 1)
    return total

if __name__ == "__main__":
    pass

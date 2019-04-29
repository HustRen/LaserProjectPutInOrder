# coding:utf-8
import os
import math

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

def file_name(file_dir, filter, anstype):
    """获取指定文件夹下所有指定文件的文件名
    Arguments:
        file_dir {str} -- 文件夹路径
        filter {str} -- 所需文件后缀名。如：'.png'
        anstype {bool} -- 结果是否包含路径 true包含， false不包含
    Returns:
        [list] -- 文件名列表，包含后缀但不含路径
    """
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == filter:
                if anstype == True:
                    L.append(os.path.join(root, file))
                else:
                    L.append(file)           
    return L

def traversalDir_FirstDir(path):
    """获取路径下所有文件夹名称
    Arguments:
        path {str} -- 文件夹路径   
    Returns:
        [list] -- 该文件夹下所有文件的文件名列表
    """
     # 定义一个列表，用来存储结果
    list = []
    # 判断路径是否存在
    if (os.path.exists(path)):
        # 获取该目录下的所有文件或文件夹目录
        files = os.listdir(path)
        for file in files:
            # 得到该文件下所有目录的路径
            m = os.path.join(path, file)
            # 判断该路径下是否是文件夹
            if (os.path.isdir(m)):
                h = os.path.split(m)
                list.append(h[1])
    return list

def GetSatrtXYFromPolar(imageshape, targetshape, radius, angle):
    imgRow = imageshape[0]
    imgCol = imageshape[1]
    targetRow = targetshape[0]
    targetCol = targetshape[1]

    imgCenRow = imgRow / 2
    imgCenCol = imgCol / 2
    targetCenRow = targetRow / 2
    targetCenCol = targetCol / 2
    startRow  = int(min(max(imgCenRow - radius * math.sin((angle / 180) * math.pi) - targetCenRow, 0), imgRow - targetRow))
    startCol  = int(min(max(imgCenCol + radius * math.cos((angle / 180) * math.pi) - targetCenCol, 0), imgCol - targetCol))
    return [startRow, startCol]


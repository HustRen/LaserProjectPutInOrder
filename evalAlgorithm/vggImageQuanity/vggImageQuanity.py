import sys,os
from math import pow

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics.pairwise import (cosine_distances, cosine_similarity,
                                      euclidean_distances)

import utils
import vgg19

sys.path.append(os.path.dirname(__file__) + os.sep + '../../lib')
from imageQualityAlgor import MFSIM, PSNR, SSIM, WFSIM
from tools import file_name, mkdir, traversalDir_FirstDir


def loss(temp, dist):
    x = temp.reshape(1,-1)
    y = dist.reshape(1,-1)
    # cos = cosine_distances(x,y)
    cos = cosine_similarity(x,y)
    # loss = np.mean((2*x*y + 0.0001) / (x**2 + y**2 + 0.0001))
    loss = cos[0][0]
    return loss

def euclideanDistance(temp, dist):
    x = temp.reshape(1,-1)
    y = dist.reshape(1,-1)
    cos = euclidean_distances(x,y)
    loss = cos[0][0]
    return loss

def vggImageQuanity(tempImage, distImage):
    with tf.Session() as sess:
        images = tf.placeholder("float", [2, 224, 224, 3])
        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        img1 = utils.resizeImg(tempImage)
        img2 = utils.resizeImg(distImage)
        batch1 = img1.reshape((1, 224, 224, 3))
        batch2 = img2.reshape((1, 224, 224, 3))
        batch = np.concatenate((batch1, batch2), 0)
        feed_dict = {images: batch}

        c1 = sess.run(vgg.pool1, feed_dict=feed_dict)
        loss1 = loss(c1[0], c1[1])
        
        c2 = sess.run(vgg.pool2, feed_dict=feed_dict)
        loss2 = loss(c2[0], c2[1])

        c3 = sess.run(vgg.pool3, feed_dict=feed_dict)
        loss3 = loss(c3[0], c3[1])

        c4 = sess.run(vgg.pool4, feed_dict=feed_dict)
        loss4 = loss(c4[0], c4[1])

        c5 = sess.run(vgg.pool5, feed_dict=feed_dict)
        loss5 = loss(c5[0], c5[1])

        total = pow(loss1, 1.1) * pow(loss2,1.05) * pow(loss3,1.0) * pow(loss4,0.95) * pow(loss5,0.9)
    return total

   
      
# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
def mian():
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [2, 224, 224, 3])
            vgg = vgg19.Vgg19()
            with tf.name_scope("content_vgg"):
                vgg.build(images)
            #names = ['20_81_67', '20_104_93', '20_141_101', '20_190_114']
            # names = ['20_141_101', '30_93_75', '40_92_87', '50_109_129', '60_133_147']
            names = ['0', '20', '40', '60', '80', '100']
            levels = ['level0', 'level2', 'level5', 'level1', 'level4', 'level7', 'level8']
            #levels = ['airplane_007']
            airplane = 'airplane_007'
            floder = 'D:/LaserData/background/1024X1024/batch2/' + airplane
            # names = file_name(floder, '.png', False)
            # names = file_name('D:/LaserData/ans/template_0_0', '.png', False)
            lens = len(names) * len(levels)
            c1s = np.zeros(lens, dtype=float)
            c2s = np.zeros(lens, dtype=float)
            c3s = np.zeros(lens, dtype=float)
            c4s = np.zeros(lens, dtype=float)
            c5s = np.zeros(lens, dtype=float)
            totals = np.zeros(lens, dtype=float)
            mfsims = np.zeros(lens, dtype=float)
            newsim = np.zeros(lens, dtype=float)
            k = 0
            for j in range(0, len(levels)):
                level = levels[j]
                for i in range(0, len(names)):
                    name = names[i] 
                    # temp = 'D:/LaserData/ans/scr/' + airplane + '.jpg'
                    # dist = floder + '/' + name
                    temp = 'D:/LaserData/ans/template/template_0_' + name + '.png'
                    dist = 'D:/LaserData/ans/' + level + '/' + level + '_0_' + name + '.png'
                    # temp = 'D:/LaserData/ans/template/template_0_0.png'
                    # dist = 'D:/LaserData/ans/' + level + '/' +  name
                    tempImage = cv2.imread(temp, cv2.IMREAD_GRAYSCALE)
                    distImage = cv2.imread(dist, cv2.IMREAD_GRAYSCALE) 

                    img1 = utils.load_image(temp)
                    img2 = utils.load_image(dist)

                    batch1 = img1.reshape((1, 224, 224, 3))
                    batch2 = img2.reshape((1, 224, 224, 3))

                    batch = np.concatenate((batch1, batch2), 0)
                    feed_dict = {images: batch}

                    c1 = sess.run(vgg.pool1, feed_dict=feed_dict)
                    loss1 = loss(c1[0], c1[1])
                    c1s[k] = loss1
                    
                    c2 = sess.run(vgg.pool2, feed_dict=feed_dict)
                    loss2 = loss(c2[0], c2[1])
                    c2s[k] = loss2

                    c3 = sess.run(vgg.pool3, feed_dict=feed_dict)
                    loss3 = loss(c3[0], c3[1])
                    c3s[k] = loss3

                    c4 = sess.run(vgg.pool4, feed_dict=feed_dict)
                    loss4 = loss(c4[0], c4[1])
                    c4s[k] = loss4

                    c5 = sess.run(vgg.pool5, feed_dict=feed_dict)
                    loss5 = loss(c5[0], c5[1])
                    c5s[k] = loss5

                    total = pow(loss1, 1.1) * pow(loss2,1.05) * pow(loss3,1.0) * pow(loss4,0.95) * pow(loss5,0.9)
                    totals[k] = total

                    a = MFSIM(tempImage, distImage)
                    mfsims[k] = a
                    newsim[k] = a * loss5
                    print('%s_0_%s c1: %f  c2: %f  c3: %f  c4: %f  c5: %f  MFSIM: %f'%(level, name, loss1, loss2, loss3, loss4, loss5, a))
                    k = k + 1
                    #print('%s_%s c1: %f  c2: %f  c3: %f  c4: %f  c5: %f  ans: %f MFSIM: %f'%(airplane, name, loss1, loss2, loss3, loss4, loss5, total, a))

            plt.figure(1)
            plt.plot(totals, label='total')
            plt.plot(mfsims, label='mfsim')
            plt.plot(newsim, label='newsim')
            plt.legend(loc="best")
            plt.xticks(range(0, lens))
            plt.grid()
            plt.show()
            # prob = sess.run(vgg.prob, feed_dict=feed_dict)
            # print(prob)c5
            # utils.print_prob(prob[0], './synset.txt')
            # utils.print_prob(prob[1], './synset.txt')


if __name__ == "__main__":
    mian()
    # disstanceCompare()

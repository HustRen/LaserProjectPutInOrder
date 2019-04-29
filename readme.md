# 基本介绍
## 一、evalAlgorithm文件夹
### NRImageQuanity文件夹
这是无参考激光干扰图像质量评价的相关代码，暂时先不管
### vggImageQuanity文件夹
这是基于VGG卷积网络的激光干扰图像质量评价算法
## 二、lib文件夹
1. lib文件夹下的imageQualityAlgor.py文件包含常用的全参考图像质量评价指标和激光干扰图像质量评价算法
    *  WMS_SSIM、WFSIM、MFSIM算法原理见钱方的《激光对光电系统图像干扰的效果评估》
    * SSIM是一种经典的图像质量评价算法，网上有很多博客，建议看一下原文
2. lib文件夹下的simCalss.py是图像仿真的相关代码，暂时不用
3. lib文件夹下的tools.py是一些常用的工具函数，需要是查看
# 目前任务
主要了解lib文件夹下的imageQualityAlgor.py文件中的算法及其原理，对图像质量评价和激光干扰图像质量评价的概念有一个基本了解
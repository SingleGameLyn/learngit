# coding:utf-8
# !/usr/bin/env python

# test git update
# commit and push
import cv2
from numpy import *
import numpy as np
# import PIL
import time
import pandas as pd

start = time.clock()

# from PIL import Image
np.set_printoptions(threshold='nan')

screenLevels = 255.0


def yuv_import(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    blk_size = prod(dims) * 3 / 2
    fp.seek(blk_size * startfrm, 0)
    Y = []
    U = []
    V = []
    # print dims[0]
    # print dims[1]
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    # print d00
    # print d01
    Yt = zeros((dims[0], dims[1]), uint8, 'C')
    Ut = zeros((d00, d01), uint8, 'C')
    Vt = zeros((d00, d01), uint8, 'C')

    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                # print m, n
                Yt[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                Ut[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                Vt[m, n] = ord(fp.read(1))
        Y = Y + [Yt]
        U = U + [Ut]
        V = V + [Vt]
    fp.close()
    return (Y, U, V)


if __name__ == '__main__':
    width = 1920
    height = 1080
    # url_NTT = '/home/d066/Videos/NTT_repeat_20frms.yuv'

    url_ori = '/home/lx/Videos/CrowdRun_1080p50.yuv'
    # url_NTT = '/home/lx/Videos/NTT_10frms.yuv'
    # url_NTT = '/home/lx/Videos/NTT_1080p50_10Mbps_8bit.yuv'
    url_NTT = '/home/lx/Videos/NTT_1080p50_10Mbps_8bit_500of1010.yuv'

    iters = 3
    datas_ori = []
    datas_NTT = []
    costs = []

# 读取ori的 iters 帧图像
    Ori_Y = []
    Ori_U = []
    Ori_V = []
    for i in range(iters):
        data_ori = yuv_import(url_ori, (height, width), 1, i)  # 读一帧,从第i帧开始,每次读一帧读YUV,是一个3行一列的矩阵
        Ori_Y.append(data_ori[0][0])    # 读Y
        Ori_U.append(data_ori[1][0])    # 读U
        Ori_V.append(data_ori[2][0])    # 读V
        print 'This is {}th frm -- Ori'.format(i)

    Ori_Y_arr = np.array(Ori_Y)  # Ori_Y_arr: Ori图像数据Y, uint8 类型
    print 'The shape of Ori_Y_arr is :', Ori_Y_arr.shape       # (3, 1080, 1920)  3表示帧数
    Ori_U_arr = np.array(Ori_U)
    Ori_V_arr = np.array(Ori_V)

    # data_ori_1st_frm = yuv_import(url_ori, (height, width), 1, 0)  # data_ori_1st_frm：ori的原始图像， uint8类型
    # # XX_arr = np.array(data_ori_1st_frm)
    # Ori_Y_arr = data_ori_1st_frm[0][0].astype(int8)  # arr_Y：ori的Y分量，int8类型
    # Ori_U_arr = data_ori_1st_frm[1][0].astype(int8)
    # Ori_V_arr = data_ori_1st_frm[2][0].astype(int8)

    Ori_int8_Y = []
    Ori_int8_U = []
    Ori_int8_V = []
    for i in range(iters):
        arr2_Y = Ori_Y_arr[i].astype(int8)   # 将每一帧的数据由 uint8 转成 int8
        Ori_int8_Y.append(arr2_Y)
        arr2_U = Ori_U_arr[i].astype(int8)
        Ori_int8_U.append(arr2_U)
        arr2_V = Ori_V_arr[i].astype(int8)
        Ori_int8_V.append(arr2_V)
        print 'change NTT {}th frm -- Ori'.format(i)

    Ori_int8_Y_arr = np.array(Ori_int8_Y)  # Ori int8类型数据
    print 'The shape of Ori_int8_Y_arr is :', Ori_int8_Y_arr.shape   # (2, 1080, 1920) 同样2表示两帧视频
    Ori_int8_U_arr = np.array(Ori_int8_U)  # Ori int8类型数据
    Ori_int8_V_arr = np.array(Ori_int8_V)  # Ori int8类型数据
    print Ori_int8_Y_arr.shape  # (10, 1080, 1920)

# 读取NTT的 iters 帧图像
#     NTT_Y = []
#     NTT_U = []
#     NTT_V = []
#     for i in range(iters):
#         data_NTT = yuv_import(url_NTT, (height, width), 1, i)
#         NTT_Y.append(data_NTT[0][0])
#         NTT_U.append(data_NTT[1][0])
#         NTT_V.append(data_NTT[2][0])
#         # cv2.imshow("sohow{}".format(i), YYx[i])
#         print 'This is {}th frm -- NTT'.format(i)
#
#     NTT_Y_arr = np.array(NTT_Y)  # YYx_Y_arr： NTT图像数据， uint8 类型
#     NTT_U_arr = np.array(NTT_U)
#     NTT_V_arr = np.array(NTT_V)
#     print 'The shape of NTT_Y_arr.shape is :', NTT_Y_arr.shape   # (2, 1080, 1920) 同样2表示两帧视频
#
#     # print NTT_Y_arr.shape
#
#     NTT_int8_Y = []
#     NTT_int8_U = []
#     NTT_int8_V = []
#     for i in range(iters):
#         arr2_Y = NTT_Y_arr[i].astype(int8)
#         NTT_int8_Y.append(arr2_Y)
#         arr2_U = NTT_U_arr[i].astype(int8)
#         NTT_int8_U.append(arr2_U)
#         arr2_V = NTT_V_arr[i].astype(int8)
#         NTT_int8_V.append(arr2_V)
#         print 'change NTT {}th frm'.format(i)
#
#     NTT_int8_Y_arr = np.array(NTT_int8_Y)  # NTT int8类型数据
#     NTT_int8_U_arr = np.array(NTT_int8_U)  # NTT int8类型数据
#     NTT_int8_V_arr = np.array(NTT_int8_V)  # NTT int8类型数据
#     print NTT_int8_Y_arr.shape  # (10, 1080, 1920)

# SSIM
#     SSIM  用 uint8 类型的数据
#     Y
#     ux_Y = np.mean(YYx_Y_arr)
#     print ux_Y
#     uy_Y = np.mean()
    ux_Y = []
    for i in range(iters):
        temp = np.mean(Ori_Y_arr[i])    # 第i帧的均值
        ux_Y.append(temp)
    print type(ux_Y)    # <type 'list'>
    ux_Y_arr = np.array(ux_Y)
    print ux_Y_arr  # [106.14581983 106.30893615 106.2186304 ]
    print ux_Y_arr[0]   # 106.14581983
    print ux_Y_arr.shape    # (3,)
    print ux_Y_arr[0].shape     # ()
    ux_Y_arr = ux_Y_arr[:, np.newaxis]
    print ux_Y_arr.shape  # (3,1)
    print ux_Y_arr[0]   # [106.14581983]
    print ux_Y_arr[1]   # [106.30893615]
    print ux_Y_arr[2]   # [106.2186304]

# calculate PSNR
#
# # cost Y
#     ccc_cost_Y = []
#     for k in range(iters):
#         temp = Ori_int8_Y_arr[k] - NTT_int8_Y_arr[k]
#
#         # temp = Ori_Y_arr - NTT_int8_Y_arr[k]
#
#         temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#         temp2 = np.array(temp2)
#         temp3 = np.sum(temp2)
#         ccc = (1.0 / (width * height)) * temp3
#         ccc_cost_Y.append(ccc)
#         print 'calculate {}th frm -- Y'.format(k)
#
#     print ccc_cost_Y  # MSE 矩阵 <type 'list'>
#     ccc_cost_Y_arr = np.array(ccc_cost_Y)
#     print ccc_cost_Y_arr  # <type 'numpy.ndarray'>
#     print ccc_cost_Y_arr.shape  # (40,)
#
#     ccc_psnr_Y_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_Y_arr))  # psnr 矩阵
#     print 'ccc_psnr_Y_arr = ', ccc_psnr_Y_arr
#
# # SSIM
#     # SSIM  用 uint8 类型的数据
#     # Y
#     # ux_Y = np.mean(YYx_Y_arr)
#     # print ux_Y
#     # uy_Y = np.mean()
#     ux_Y = []
#     for i in range(iters):
#         temp = np.mean(Ori_Y_arr[i])    # 第i帧的均值
#         ux_Y.append(temp)
#
#
#
#     # data_ori_1st_frm：ori的原始图像， uint8类型
#     # YYx_Y_arr： NTT图像数据， uint8 类型
#     # print data_ori_1st_frm[0][0] >= 0   # 全部True
#     # print arr_Y >= 0    # 有True, 有False
#     # k = []
#     # for i in range(iters):
#     #     data_arr = np.array(data_ori_1st_frm[i][0])
#     #     ux_Y = np.mean(data_arr)
#     #     k.append(ux_Y)
#     # print k
#     #
#
# # cost U
#     ccc_cost_U = []
#     for k in range(iters):
#         temp = Ori_int8_U_arr[k] - NTT_int8_U_arr[k]
#         temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#         temp2 = np.array(temp2)
#         temp3 = np.sum(temp2)
#         ccc = (1.0 / (width * height)) * temp3
#         ccc_cost_U.append(ccc)
#         print 'calculate {}th frm -- U'.format(k)
#
#     print ccc_cost_U   # MSE 矩阵 <type 'list'>
#     ccc_cost_U_arr = np.array(ccc_cost_U)
#     print ccc_cost_U_arr  # <type 'numpy.ndarray'>
#     print ccc_cost_U_arr.shape    # (40,)
#
#     ccc_psnr_U_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_U_arr))   # psnr 矩阵
#     print 'ccc_psnr_U_arr = ', ccc_psnr_U_arr
#
# # cost V
#     ccc_cost_V = []
#     for k in range(iters):
#         temp = Ori_int8_V_arr[k] - NTT_int8_V_arr[k]
#         temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#         temp2 = np.array(temp2)
#         temp3 = np.sum(temp2)
#         ccc = (1.0 / (width * height)) * temp3
#         ccc_cost_V.append(ccc)
#         print 'calculate {}th frm -- U'.format(k)
#
#     print ccc_cost_V   # MSE 矩阵 <type 'list'>
#     ccc_cost_V_arr = np.array(ccc_cost_V)
#     print ccc_cost_V_arr  # <type 'numpy.ndarray'>
#     print ccc_cost_V_arr.shape    # (40,)
#
#     ccc_psnr_V_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_V_arr))   # psnr 矩阵
#     print 'ccc_psnr_V_arr = ', ccc_psnr_V_arr

# 输出 csv 文件
#     test_Y = pd.Series(ccc_psnr_Y_arr)
#     test_U = pd.Series(ccc_psnr_U_arr)
#     test_V = pd.Series(ccc_psnr_V_arr)
#     test = pd.DataFrame({'Y_psnr': test_Y, 'U_psnr': test_U, 'V_psnr': test_V})
#     test = test[['Y_psnr', 'U_psnr', 'V_psnr']]     # 按照YUV的顺序进行输出
#     test.to_csv('/home/lx/Videos/NTT500of1010.csv')
    print 'start show...'

    # print 'uint8', np.mean(data_ori_1st_frm[0][0])   # data_ori_1st_frm：ori的原始图像Y分量， uint8类型
    # print 'int8', np.mean(arr_Y)                    # arr_Y: ori的原始图像Y分量， uint8类型
    # ux_Y = np.mean(YYx_Y_arr)
    # print ux_Y

    cv2.waitKey(0)
end = time.clock()
print('Running time: %s Seconds' % (end - start))

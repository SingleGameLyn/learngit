#coding:utf-8
#!/usr/bin/env python

# test git update
# commit and push
import cv2
from numpy import *
import numpy as np
# import PIL
import time
import pandas as pd

start =time.clock()

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
    # url_ori = '/home/d066/Videos/CrowdRun_1080p50-0_10frms.yuv'
    # # url_NTT = '/home/d066/Videos/NTT_10frms.yuv'
    # url_NTT = '/home/d066/Videos/NTT_repeat_20frms.yuv'

    url_ori = '/home/d066/Videos/CrowdRun_1080p50-0_10frms.yuv'
    # url_NTT = '/home/lx/Videos/NTT_10frms.yuv'
    url_NTT = '/home/d066/Videos/NTT_1080p50_10Mbps_8bit.yuv'

    iters = 3
    datas_ori = []
    datas_NTT = []
    costs = []

# 读取ori的第一帧图像
    data_ori_1st_frm = yuv_import(url_ori, (height, width), 1, 0)       # data_ori_1st_frm：ori的原始图像， uint8类型
    # XX_arr = np.array(data_ori_1st_frm)
    arr_Y = data_ori_1st_frm[0][0].astype(int8)     # arr_Y：ori的Y分量，int8类型
    arr_U = data_ori_1st_frm[1][0].astype(int8)
    arr_V = data_ori_1st_frm[2][0].astype(int8)

# 一次性读取NTT十帧图像
    YYx_Y = []
    YYx_U = []
    YYx_V = []
    for i in range(iters):
        data_NTT_10frms = yuv_import(url_NTT, (height, width), 1, i)
        YYx_Y.append(data_NTT_10frms[0][0])
        YYx_U.append(data_NTT_10frms[1][0])
        YYx_V.append(data_NTT_10frms[2][0])
        # cv2.imshow("sohow{}".format(i), YYx[i])
        print 'This is {}th frm'.format(i)

    YYx_Y_arr = np.array(YYx_Y)            # YYx_Y_arr： NTT图像数据， uint8 类型
    YYx_U_arr = np.array(YYx_U)
    YYx_V_arr = np.array(YYx_V)
    print YYx_Y_arr.shape     # (10, 1080, 1920)
    print arr_Y.shape

    arr2s_Y = []
    arr2s_U = []
    arr2s_V = []
    for i in range(iters):
        arr2_Y = YYx_Y_arr[i].astype(int8)
        arr2s_Y.append(arr2_Y)
        arr2_U = YYx_U_arr[i].astype(int8)
        arr2s_U.append(arr2_U)
        arr2_V = YYx_V_arr[i].astype(int8)
        arr2s_V.append(arr2_V)
        print 'change NTT {}th frm'.format(i)

    arr2_Y_arr = np.array(arr2s_Y)  # NTT int8类型数据
    arr2_U_arr = np.array(arr2s_U)  # NTT int8类型数据
    arr2_V_arr = np.array(arr2s_V)  # NTT int8类型数据
    print arr2_Y_arr.shape    # (10, 1080, 1920)

# cost Y
    ccc_cost_Y = []
    for k in range(iters):
        temp = arr_Y - arr2_Y_arr[k]
        temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
        temp2 = np.array(temp2)
        temp3 = np.sum(temp2)
        ccc = (1.0 / (width * height)) * temp3
        ccc_cost_Y.append(ccc)
        print 'calculate {}th frm -- Y'.format(k)

    print ccc_cost_Y   # MSE 矩阵 <type 'list'>
    ccc_cost_Y_arr = np.array(ccc_cost_Y)
    print ccc_cost_Y_arr  # <type 'numpy.ndarray'>
    print ccc_cost_Y_arr.shape    # (40,)

    ccc_psnr_Y_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_Y_arr))   # psnr 矩阵
    print 'ccc_psnr_Y_arr = ', ccc_psnr_Y_arr
    # SSIM  用 uint8 类型的数据
    # Y
    # ux_Y = np.mean(YYx_Y_arr)
    # print ux_Y
    # uy_Y = np.mean()

    # data_ori_1st_frm：ori的原始图像， uint8类型
    # YYx_Y_arr： NTT图像数据， uint8 类型
    

# cost U
    ccc_cost_U = []
    for k in range(iters):
        temp = arr_U - arr2_U_arr[k]
        temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
        temp2 = np.array(temp2)
        temp3 = np.sum(temp2)
        ccc = (1.0 / (width * height)) * temp3
        ccc_cost_U.append(ccc)
        print 'calculate {}th frm -- U'.format(k)

    print ccc_cost_U   # MSE 矩阵 <type 'list'>
    ccc_cost_U_arr = np.array(ccc_cost_U)
    print ccc_cost_U_arr  # <type 'numpy.ndarray'>
    print ccc_cost_U_arr.shape    # (40,)

    ccc_psnr_U_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_U_arr))   # psnr 矩阵
    print 'ccc_psnr_U_arr = ', ccc_psnr_U_arr

# cost V
    ccc_cost_V = []
    for k in range(iters):
        temp = arr_V - arr2_V_arr[k]
        temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
        temp2 = np.array(temp2)
        temp3 = np.sum(temp2)
        ccc = (1.0 / (width * height)) * temp3
        ccc_cost_V.append(ccc)
        print 'calculate {}th frm -- U'.format(k)

    print ccc_cost_V   # MSE 矩阵 <type 'list'>
    ccc_cost_V_arr = np.array(ccc_cost_V)
    print ccc_cost_V_arr  # <type 'numpy.ndarray'>
    print ccc_cost_V_arr.shape    # (40,)

    ccc_psnr_V_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_V_arr))   # psnr 矩阵
    print 'ccc_psnr_V_arr = ', ccc_psnr_V_arr

# 输出 csv 文件
    test_Y = pd.Series(ccc_psnr_Y_arr)
    test_U = pd.Series(ccc_psnr_U_arr)
    test_V = pd.Series(ccc_psnr_V_arr)
    test = pd.DataFrame({'Y_psnr': test_Y, 'U_psnr': test_U, 'V_psnr': test_V})
    test = test[['Y_psnr', 'U_psnr', 'V_psnr']]     # 按照YUV的顺序进行输出
    # test.to_csv('/home/lx/Videos/NTT555.csv')
    print 'start show...'

    print 'uint8', np.mean(data_ori_1st_frm[0][0])   # data_ori_1st_frm：ori的原始图像Y分量， uint8类型
    print 'int8', np.mean(arr_Y)                    # arr_Y: ori的原始图像Y分量， uint8类型
    ux_Y = np.mean(YYx_Y_arr)
    print ux_Y

    cv2.waitKey(0)
end = time.clock()
print('Running time: %s Seconds' % (end-start))

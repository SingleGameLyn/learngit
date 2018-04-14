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
    fp.seek(int(blk_size) * startfrm, 0)
    Y = []
    U = []
    V = []
    d00 = dims[0] // 2
    d01 = dims[1] // 2
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

    yuv_Y_arr = np.array(Y)  # yuv_Y_arr: Ori图像数据Y, uint8 类型
    yuv_U_arr = np.array(U)
    yuv_V_arr = np.array(V)

    yuv_int8_Y_arr = array(ndarray.tolist(yuv_Y_arr))  # (3, 1080, 1920) Ori图像数据 int64类型
    yuv_int8_U_arr = array(ndarray.tolist(yuv_U_arr))
    yuv_int8_V_arr = array(ndarray.tolist(yuv_V_arr))

    return yuv_int8_Y_arr, yuv_int8_U_arr, yuv_int8_V_arr

    # return (Y, U, V)


# def yuv_readfrms_int64(filename, (height, width), frms):
#     yuv_Y = []
#     yuv_U = []
#     yuv_V = []
#     for i in range(frms):
#         data = yuv_import(filename, (height, width), 1, i)  # 读一帧,从第i帧开始,每次读一帧读YUV,是一个3行一列的矩阵
#         yuv_Y.append(data[0][0])    # 读Y
#         yuv_U.append(data[1][0])    # 读U
#         yuv_V.append(data[2][0])    # 读V
#         print ('This is {}th frm -- {}'.format(i, filename))
#
#     time100 = time.clock()
#
#     yuv_Y_arr = np.array(yuv_Y)  # yuv_Y_arr: Ori图像数据Y, uint8 类型
#     yuv_U_arr = np.array(yuv_U)
#     yuv_V_arr = np.array(yuv_V)
#
#     yuv_int8_Y_arr = array(ndarray.tolist(yuv_Y_arr))  # (3, 1080, 1920) Ori图像数据 int64类型
#     yuv_int8_U_arr = array(ndarray.tolist(yuv_U_arr))
#     yuv_int8_V_arr = array(ndarray.tolist(yuv_V_arr))
#
#     time101 = time.clock()
#     print ('time12 - time11 = ', time101 - time100)
#
#     return yuv_int8_Y_arr, yuv_int8_U_arr, yuv_int8_V_arr


def cal_ssim(Ori_int64_Y_arr, NTT_int64_Y_arr, (height, width)):
    # 计算每一帧的均值ux
    ux_Y = np.mean(Ori_int64_Y_arr[0])  # 第i帧的均值
    uy_Y = np.mean(NTT_int64_Y_arr[0])

    # 尝试 numpy的矩阵操作
    # Ori的每一帧方差
    meanx_arr = np.ones((height, width))  # 构造所有元素都等于均值的矩阵
    for i in range(len(meanx_arr)):  # 1080次
        for j in range(len(meanx_arr[0])):  # 1920次
            meanx_arr[i][j] = ux_Y

    temp1 = Ori_int64_Y_arr[0] - meanx_arr
    temp2 = [[temp1[i][j] ** 2 for j in range(len(temp1[i]))] for i in range(len(temp1))]
    temp3 = np.sum(temp2)
    sigma2x_np = (1.0 / ((height * width) - 1)) * temp3  # 当前帧的方差
    sigma_x_np = sqrt(sigma2x_np)  # Ori 标准差  当前帧标准差

    # NTT的每一帧方差
    meany_arr = np.ones((height, width))
    for i in range(len(meany_arr)):
        for j in range(len(meany_arr[0])):
            meany_arr[i][j] = uy_Y

    temp1 = NTT_int64_Y_arr[0] - meany_arr
    temp2 = [[temp1[i][j] ** 2 for j in range(len(temp1[i]))] for i in range(len(temp1))]
    temp3 = np.sum(temp2)
    sigma2y_np = (1.0 / (height * width)) * temp3

    sigma_y_np = sqrt(sigma2y_np)  # NTT 标准差

    # multi_xy = []  # 存放计算 (X(i,j) - ux)*(Y(i,j) - uy)得到的矩阵
    multi_xy = (Ori_int64_Y_arr[0] - meanx_arr) * (NTT_int64_Y_arr[0] - meany_arr)

    temp_arr_1frm_sum = np.sum(multi_xy)  # 第i帧的(X(i,j) - ux)*(Y(i,j) - uy)求和
    conv_sigma_xy = (1.0 / ((width * height) - 1)) * temp_arr_1frm_sum  # 协方差计算公式

    K1 = 0.01
    K2 = 0.03
    LL = 255
    C1 = (K1 * LL) ** 2
    C2 = (K2 * LL) ** 2
    C3 = C2 / 2
    # L(x,y)
    Lxy = (2 * ux_Y * uy_Y + C1) / ((ux_Y) ** 2 + (uy_Y) ** 2 + C1)
    # print ('L(x,y)')
    # print (Lxy)

    # C(x,y)
    Cxy = (2 * sigma_x_np * sigma_y_np + C2) / (sigma2x_np + sigma2y_np + C2)
    # print ('C(x,y)')
    # print (Cxy)

    # S(x,y)
    Sxy = (conv_sigma_xy + C3) / (sigma_x_np * sigma_y_np + C3)
    # print ('S(x,y)')
    # print (Sxy)

    # SSIM score
    SSIM = Lxy * Cxy * Sxy
    # print ('SSIM')
    # print (SSIM)

    return SSIM


def cal_psnr(Ori_int64_Y_arr, NTT_int64_Y_arr, (height, width)):
    temp = Ori_int64_Y_arr[0] - NTT_int64_Y_arr[0]
    temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
    temp2 = np.array(temp2)
    temp3 = np.sum(temp2)
    mse = (1.0 / (width * height)) * temp3

    psnr_Y = 20 * np.log10(255 / np.sqrt(mse))  # psnr 矩阵

    return psnr_Y


if __name__ == '__main__':
    width = 1920
    height = 1080

    url_ori = '/home/lx/Videos/CrowdRun_1080p50.yuv'
    url_NTT = '/home/lx/Videos/NTT_1080p50_10Mbps_8bit_500of1010.yuv'

    iters = 500

    for i in range(iters):
        print ('This is {}th calculation'.format(i))
        # 读取ori的 iters 帧图像
        Ori_int64_YUV_arr = yuv_import(url_ori, (height, width), 1, i)

        Ori_int64_Y_arr = Ori_int64_YUV_arr[0]      # Ori 图像Y分量数据 int64类型
        Ori_int64_U_arr = Ori_int64_YUV_arr[1]      # Ori 图像U分量数据 int64类型
        Ori_int64_V_arr = Ori_int64_YUV_arr[2]      # Ori 图像V分量数据 int64类型

        # 读取NTT的 iters 帧图像
        NTT_int64_YUV_arr = yuv_import(url_NTT, (height, width),1, i)

        NTT_int64_Y_arr = NTT_int64_YUV_arr[0]
        NTT_int64_U_arr = NTT_int64_YUV_arr[1]
        NTT_int64_V_arr = NTT_int64_YUV_arr[2]

        # calculate SSIM******************************************************************************************
        SSIM = cal_ssim(Ori_int64_Y_arr, NTT_int64_Y_arr,(height, width))

        # calculate PSNR*******************************************************************************************
        # #cost Y
        psnr_Y = cal_psnr(Ori_int64_Y_arr, NTT_int64_Y_arr, (height, width))
        # cost U
        psnr_U = cal_psnr(Ori_int64_U_arr, NTT_int64_U_arr, (height, width))
        # cost V
        psnr_V = cal_psnr(Ori_int64_V_arr, NTT_int64_V_arr, (height, width))

        # 输出 csv 文件
        name = ['Y_psnr', 'U_psnr', 'V_psnr', '', 'SSIM']

        # df = pd.DataFrame([[psnr_Y, psnr_U, psnr_V, None, SSIM]], columns=name)
        # df.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test_HHH222222.csv')

        if i == 0:
            df = pd.DataFrame([[psnr_Y, psnr_U, psnr_V, None, SSIM]], columns=name)
        else:
            df2 = pd.DataFrame([[psnr_Y, psnr_U, psnr_V, None, SSIM]], columns=name)
            df = df.append(df2, ignore_index=True)

        df.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test_HHH.csv')

        # if i == 0:
        #     df = pd.DataFrame([test_Y, test_U, test_V, test_none, test_SSIM], columns=name)
        #     df.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test_HHH.csv')
        # else:
        #     df2 = pd.DataFrame([test_Y, test_U, test_V, test_none, test_SSIM], columns=name)
        #     df = df.append(df2, ignore_index=True)
        #     df.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test_HHH2.csv')

    cv2.waitKey(0)
end = time.clock()
print('Running time: %s Seconds' % (end - start))

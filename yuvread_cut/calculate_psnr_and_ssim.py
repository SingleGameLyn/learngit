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


def yuv_readfrms_int64(filename, (height, width), frms):
    yuv_Y = []
    yuv_U = []
    yuv_V = []
    for i in range(frms):
        data = yuv_import(filename, (height, width), 1, i)  # 读一帧,从第i帧开始,每次读一帧读YUV,是一个3行一列的矩阵
        yuv_Y.append(data[0][0])    # 读Y
        yuv_U.append(data[1][0])    # 读U
        yuv_V.append(data[2][0])    # 读V
        print ('This is {}th frm -- {}'.format(i, filename))

    time100 = time.clock()

    yuv_Y_arr = np.array(yuv_Y)  # yuv_Y_arr: Ori图像数据Y, uint8 类型
    yuv_U_arr = np.array(yuv_U)
    yuv_V_arr = np.array(yuv_V)

    # Ori_int8_Y_list = ndarray.tolist(yuv_Y_arr)     # Ori   转成list 过渡
    # Ori_int8_U_list = ndarray.tolist(yuv_U_arr)
    # Ori_int8_V_list = ndarray.tolist(yuv_V_arr)
    #
    # Ori_int8_Y_arr = array(Ori_int8_Y_list)     # (3, 1080, 1920) Ori图像数据 int64类型
    # Ori_int8_U_arr = array(Ori_int8_U_list)
    # Ori_int8_V_arr = array(Ori_int8_V_list)

    yuv_int8_Y_arr = array(ndarray.tolist(yuv_Y_arr))  # (3, 1080, 1920) Ori图像数据 int64类型
    yuv_int8_U_arr = array(ndarray.tolist(yuv_U_arr))
    yuv_int8_V_arr = array(ndarray.tolist(yuv_V_arr))

    time101 = time.clock()
    print ('time12 - time11 = ', time101 - time100)

    return yuv_int8_Y_arr, yuv_int8_U_arr, yuv_int8_V_arr


# SSIM*****************************************************************************************************************
#     SSIM  用 uint8 类型的数据(错),也转成int64来处理 ,     x 表示 Ori , y 表示 NTT
def cal_ssim(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width)):
    # 计算每一帧的均值ux
    ux_Y = []  # ux
    uy_Y = []  # uy
    for i in range(iters):
        temp_Ori = np.mean(Ori_int64_Y_arr[i])  # 第i帧的均值
        ux_Y.append(temp_Ori)
        temp_NTT = np.mean(NTT_int64_Y_arr[i])
        uy_Y.append(temp_NTT)

    # print type(ux_Y)    # <type 'list'>
    ux_Y_arr = np.array(ux_Y)
    uy_Y_arr = np.array(uy_Y)
    print (ux_Y_arr)
    print('ux_Y_arr', ux_Y_arr)  # [106.14581983 106.30893615 106.2186304 ]
    # print ux_Y_arr[0]   # 106.14581983
    # print ux_Y_arr.shape    # (3,)
    # print ux_Y_arr[0].shape     # ()
    ux_Y_arr = ux_Y_arr[:, np.newaxis]  # (3,1) 表示3帧一列,一列是一列标量,不是子矩阵            Ori 每帧图像均值
    uy_Y_arr = uy_Y_arr[:, np.newaxis]  # ux_Y_arr, uy_Y_arr 表示每一帧的均帧的数组,uint8类型   NTT 每帧图像均值

    # 计算每一帧的方差sigma方
    time1 = time.clock()
    # # 尝试 循环嵌套
    # sigma2x = []
    # sigma2y = []
    # temp3 = 0
    # for k in range(iters):
    #     for i in range(height):     # 1080 行
    #         for j in range(width):     # 1920 列
    #             temp1 = Ori_Y_arr[k][i][j] - ux_Y_arr[k]
    #             temp2 = temp1 ** 2
    #             temp3 = temp3 + temp2
    #     print 'temp3', temp3
    #     temp4 = (1.0 / (height * width)) * temp3
    #     temp3 = 0
    #     sigma2x.append(temp4)
    #
    # print type(sigma2x)     # <type 'list'>
    # sigma2x = np.array(sigma2x)
    # print 'sigma2x.shape', sigma2x.shape        # (3, 1)
    # print 'sigma2x', sigma2x
    # time1_end = time.clock()
    # print '循环嵌套用时:{} seconds...'.format(time1_end - time1)
    time2 = time.clock()
    # 尝试 numpy的矩阵操作
    # Ori的每一帧方差
    meanx_arr = np.ones((iters, height, width))  # 构造所有元素都等于均值的矩阵
    for k in range(iters):
        for i in range(len(meanx_arr[0])):  # 1080次
            for j in range(len(meanx_arr[0][0])):  # 1920次
                # if meanx_arr[k][i][j] == 1:
                # print 'ux_Y_arr[{}] is {}'.format(k, ux_Y_arr[k])
                meanx_arr[k][i][j] = ux_Y_arr[k]
        print('calculate {}th frm -- meanx_arr'.format(i))
    print('**************1111111111111111111111111111111111111111111111111111111111111111**********')
    # print ux_Y_arr
    # print meanx_arr[0]         # 每一帧存放该帧的均值
    # print meanx_arr.shape   # (3, 1080, 1920)
    sigma2x_np = []
    for i in range(iters):
        temp1 = Ori_int64_Y_arr[i] - meanx_arr[i]
        print(temp1.shape)
        temp2 = [[temp1[i][j] ** 2 for j in range(len(temp1[i]))] for i in range(len(temp1))]
        # temp2 = temp1 ** 2
        temp3 = np.sum(temp2)
        temp4 = (1.0 / ((height * width) - 1)) * temp3
        sigma2x_np.append(temp4)
        print('calculate {}th frm -- sigma2x_np'.format(i))

    print(type(sigma2x_np))
    sigma2x_np = np.array(sigma2x_np)  # (3,)
    sigma2x_np = sigma2x_np[:, np.newaxis]  # (3, 1) 方差   Ori 就一列数据,每行代表一帧的方差
    print('sigma2x_np.shape', sigma2x_np.shape)
    print('sigma2x_np', sigma2x_np)
    time2_end = time.clock()
    print('numpy矩阵操作用时:{} seconds...'.format(time2_end - time2))

    sigma_x_np = sqrt(sigma2x_np)  # Ori 标准差  就一列数据,每行代表一帧的 标准差
    print(sigma_x_np)

    # NTT的每一帧方差
    meany_arr = np.ones((iters, height, width))
    for k in range(iters):
        for i in range(len(meany_arr[0])):
            for j in range(len(meany_arr[0][0])):
                meany_arr[k][i][j] = uy_Y_arr[k]
        print ('calculate {}th frm -- meany_arr'.format(i))

    sigma2y_np = []
    for i in range(iters):
        temp1 = NTT_int64_Y_arr[i] - meany_arr[i]
        print (temp1.shape)
        temp2 = [[temp1[i][j] ** 2 for j in range(len(temp1[i]))] for i in range(len(temp1))]
        temp3 = np.sum(temp2)
        temp4 = (1.0 / (height * width)) * temp3
        sigma2y_np.append(temp4)
        print ('calculate {}th frm -- sigma2y_np'.format(i))

    sigma2y_np = np.array(sigma2y_np)  # (3,)
    sigma2y_np = sigma2y_np[:, np.newaxis]  # (3, 1)    NTT 方差
    sigma_y_np = sqrt(sigma2y_np)  # NTT 标准差
    print (sigma_y_np)

    # Ori 和 NTT 对应两帧的协方差
    # ux_Y_arr_int8 = ux_Y_arr.astype(float)
    # print ux_Y_arr_int8.shape  # (3, 1)
    # uy_Y_arr_int8 = uy_Y_arr.astype(float)
    # Ori_int8_Y_arr   (2, 1080, 1920) 同样2表示两帧视频

    multi_xy = []  # 存放计算 (X(i,j) - ux)*(Y(i,j) - uy)得到的矩阵
    for i in range(iters):
        mult_temp = (Ori_int64_Y_arr[i] - meanx_arr[i]) * (NTT_int64_Y_arr[i] - meany_arr[i])
        multi_xy.append(mult_temp)
        print ('calculate {}th frm -- multi_xy'.format(i))
    multi_xy = np.array(multi_xy)  # multi_xy.shape   (2, 1, 1080, 1920)

    # temp_arr = []
    # for k in range(iters):
    #     for i in range(height):
    #         for j in range(width):
    #             temp1 = Ori_int8_Y_arr[k][i][j] - ux_Y_arr_int8[k]
    #             temp2 = NTT_int64_Y_arr[k][i][j] - ux_Y_arr_int8[k]
    #             temp3 = temp1 * temp2
    #             temp_arr.append(temp3)
    #     print 'calculate {}th frm of pre-conv'.format(k)
    # temp_arr = np.array(temp_arr)       # 计算 (X(i,j) - ux)*(Y(i,j) - uy)得到的矩阵
    # print temp_arr.shape
    # temp_arr = reshape(temp_arr, (iters, height, width))
    # print temp_arr.shape
    # print temp_arr[0].shape
    print ('****************************************')
    conv_sigma_xy = []
    for i in range(iters):
        temp_arr_1frm_sum = np.sum(multi_xy[i])  # 第i帧的(X(i,j) - ux)*(Y(i,j) - uy)求和
        temp_conv_sigma_xy = (1.0 / ((width * height) - 1)) * temp_arr_1frm_sum  # 协方差计算公式
        conv_sigma_xy.append(temp_conv_sigma_xy)
        print ('calculate {}th frm of conv_sigma_xy'.format(i))

    conv_sigma_xy_arr = np.array(conv_sigma_xy)
    conv_sigma_xy_arr = conv_sigma_xy_arr[:, np.newaxis]  # (3, 1)   多行一列,每行存放协方差
    print (conv_sigma_xy_arr)  # Ori NTT 对应帧的协方差

    K1 = 0.01
    K2 = 0.03
    LL = 255
    C1 = (K1 * LL) ** 2
    C2 = (K2 * LL) ** 2
    C3 = C2 / 2
    # L(x,y)
    L = []
    for i in range(iters):
        L_temp = (2 * ux_Y_arr[i] * uy_Y_arr[i] + C1) / ((ux_Y_arr[i]) ** 2 + (uy_Y_arr[i]) ** 2 + C1)
        L.append(L_temp)
    L = np.array(L)
    print ('L(x,y)')
    print (L)

    # C(x,y)
    C = []
    for i in range(iters):
        C_temp = (2 * sigma_x_np[i] * sigma_y_np[i] + C2) / (sigma2x_np[i] + sigma2y_np[i] + C2)
        C.append(C_temp)
    C = np.array(C)
    print ('C(x,y)')
    print (C)

    # S(x,y)
    S = []
    for i in range(iters):
        S_temp = (conv_sigma_xy_arr[i] + C3) / (sigma_x_np[i] * sigma_y_np[i] + C3)
        S.append(S_temp)
    S = np.array(S)
    print ('S(x,y)')
    print (S)

    # SSIM score
    SSIM = []
    for i in range(iters):
        SSIM_temp = L[i] * C[i] * S[i]
        SSIM.append(SSIM_temp)
    SSIM = np.array(SSIM)
    print ('SSIM')
    print (SSIM)
    SSIM = np.reshape(SSIM, (iters,))
    return SSIM


def cal_psnr(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width)):
    psnr_list = []
    for k in range(iters):
        temp = Ori_int64_Y_arr[k] - NTT_int64_Y_arr[k]
        temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
        temp2 = np.array(temp2)
        temp3 = np.sum(temp2)
        ccc = (1.0 / (width * height)) * temp3
        psnr_list.append(ccc)
        print ('calculate {}th frm -- Y'.format(k))

    print (psnr_list ) # MSE 矩阵 <type 'list'>
    psnr_list_arr = np.array(psnr_list)
    print (psnr_list_arr ) # <type 'numpy.ndarray'>
    print (psnr_list_arr.shape ) # (40,)

    psnr_Y_arr = 20 * np.log10(255 / np.sqrt(psnr_list_arr))  # psnr 矩阵
    print ('psnr_Y_arr = ', psnr_Y_arr)
    return psnr_Y_arr


if __name__ == '__main__':
    width = 1920
    height = 1080
    # url_NTT = '/home/d066/Videos/NTT_repeat_20frms.yuv'

    url_ori = '/home/lx/Videos/CrowdRun_1080p50-0_10frms.yuv'
    # url_NTT = '/home/lx/Videos/NTT_10frms.yuv'
    # url_NTT = '/home/lx/Videos/NTT_1080p50_10Mbps_8bit.yuv'
    url_NTT = '/home/lx/Videos/NTT_50frms_cut40frms.yuv'

    iters = 3
    datas_ori = []
    datas_NTT = []
    costs = []

# 读取ori的 iters 帧图像
    time11 = time.clock()
    #
    # Ori_Y = []
    # Ori_U = []
    # Ori_V = []
    # for i in range(iters):
    #     data_ori = yuv_import(url_ori, (height, width), 1, i)  # 读一帧,从第i帧开始,每次读一帧读YUV,是一个3行一列的矩阵
    #     Ori_Y.append(data_ori[0][0])    # 读Y
    #     Ori_U.append(data_ori[1][0])    # 读U
    #     Ori_V.append(data_ori[2][0])    # 读V
    #     print ('This is {}th frm -- Ori'.format(i))
    #
    #
    #
    # Ori_Y_arr = np.array(Ori_Y)  # Ori_Y_arr: Ori图像数据Y, uint8 类型
    # Ori_U_arr = np.array(Ori_U)
    # Ori_V_arr = np.array(Ori_V)
    #
    # # Ori_int8_Y_list = ndarray.tolist(Ori_Y_arr)     # Ori   转成list 过渡
    # # Ori_int8_U_list = ndarray.tolist(Ori_U_arr)
    # # Ori_int8_V_list = ndarray.tolist(Ori_V_arr)
    # #
    # # Ori_int8_Y_arr = array(Ori_int8_Y_list)     # (3, 1080, 1920) Ori图像数据 int64类型
    # # Ori_int8_U_arr = array(Ori_int8_U_list)
    # # Ori_int8_V_arr = array(Ori_int8_V_list)
    #
    # Ori_int8_Y_arr = array(ndarray.tolist(Ori_Y_arr))  # (3, 1080, 1920) Ori图像数据 int64类型
    # Ori_int8_U_arr = array(ndarray.tolist(Ori_U_arr))
    # Ori_int8_V_arr = array(ndarray.tolist(Ori_V_arr))
    Ori_int64_YUV_arr = yuv_readfrms_int64(url_ori, (height, width), iters)
    Ori_int64_Y_arr = Ori_int64_YUV_arr[0]
    Ori_int64_U_arr = Ori_int64_YUV_arr[1]
    Ori_int64_V_arr = Ori_int64_YUV_arr[2]
    time12 = time.clock()
    print ('time12 - time11 = ', time12 - time11)


# 读取NTT的 iters 帧图像
    time13 = time.clock()

    # NTT_Y = []
    # NTT_U = []
    # NTT_V = []
    # for i in range(iters):
    #     data_NTT = yuv_import(url_NTT, (height, width), 1, i)
    #     NTT_Y.append(data_NTT[0][0])
    #     NTT_U.append(data_NTT[1][0])
    #     NTT_V.append(data_NTT[2][0])
    #     # cv2.imshow("sohow{}".format(i), YYx[i])
    #     print ('This is {}th frm -- NTT'.format(i))
    #
    # NTT_Y_arr = np.array(NTT_Y)  # NTT_Y_arr： NTT图像数据， uint8 类型
    # NTT_U_arr = np.array(NTT_U)
    # NTT_V_arr = np.array(NTT_V)
    # print ('The shape of NTT_Y_arr.shape is :', NTT_Y_arr.shape )  # (2, 1080, 1920) 同样2表示两帧视频
    #
    # # NTT_int8_Y_list = ndarray.tolist(NTT_Y_arr)     # NTT   转成list 过渡
    # # NTT_int8_U_list = ndarray.tolist(NTT_U_arr)
    # # NTT_int8_V_list = ndarray.tolist(NTT_V_arr)
    # #
    # # NTT_int64_Y_arr = array(NTT_int8_Y_list)         # (3, 1080, 1920) NTT图像数据 int64类型
    # # NTT_int8_U_arr = array(NTT_int8_U_list)
    # # NTT_int8_V_arr = array(NTT_int8_V_list)
    #
    # NTT_int64_Y_arr = array(ndarray.tolist(NTT_Y_arr))         # (3, 1080, 1920) NTT图像数据 int64类型
    # NTT_int8_U_arr = array(ndarray.tolist(NTT_U_arr))
    # NTT_int8_V_arr = array(ndarray.tolist(NTT_V_arr))
    #
    # print (NTT_int64_Y_arr.shape ) # (10, 1080, 1920)
    NTT_int64_YUV_arr = yuv_readfrms_int64(url_NTT, (height, width), iters)
    NTT_int64_Y_arr = NTT_int64_YUV_arr[0]
    NTT_int64_U_arr = NTT_int64_YUV_arr[1]
    NTT_int64_V_arr = NTT_int64_YUV_arr[2]

    time14 = time.clock()
    print ('time14 - time13 = ', time14 - time13)

    SSIM = cal_ssim(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width))

# calculate PSNR*******************************************************************************************************
    time3 = time.clock()
#
# #cost Y
#     ccc_cost_Y = []
#     for k in range(iters):
#         temp = Ori_int64_Y_arr[k] - NTT_int64_Y_arr[k]
#         temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#         temp2 = np.array(temp2)
#         temp3 = np.sum(temp2)
#         ccc = (1.0 / (width * height)) * temp3
#         ccc_cost_Y.append(ccc)
#         print ('calculate {}th frm -- Y'.format(k))
#
#     print (ccc_cost_Y ) # MSE 矩阵 <type 'list'>
#     ccc_cost_Y_arr = np.array(ccc_cost_Y)
#     print (ccc_cost_Y_arr ) # <type 'numpy.ndarray'>
#     print (ccc_cost_Y_arr.shape ) # (40,)
#
#     psnr_Y_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_Y_arr))  # psnr 矩阵
#     print ('psnr_Y_arr = ', psnr_Y_arr)

    psnr_Y_arr = cal_psnr(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width))
    print ('psnr_Y_arr = ', psnr_Y_arr)
# cost U
#     ccc_cost_U = []
#     for k in range(iters):
#         temp = Ori_int64_U_arr[k] - NTT_int64_U_arr[k]
#         temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#         temp2 = np.array(temp2)
#         temp3 = np.sum(temp2)
#         ccc = (1.0 / (width * height)) * temp3
#         ccc_cost_U.append(ccc)
#         print ('calculate {}th frm -- U'.format(k))
#
#     print (ccc_cost_U  ) # MSE 矩阵 <type 'list'>
#     ccc_cost_U_arr = np.array(ccc_cost_U)
#     print (ccc_cost_U_arr ) # <type 'numpy.ndarray'>
#     print (ccc_cost_U_arr.shape )   # (40,)
#
#     ccc_psnr_U_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_U_arr))   # psnr 矩阵
#     print ('ccc_psnr_U_arr = ', ccc_psnr_U_arr)

    psnr_U_arr = cal_psnr(Ori_int64_U_arr, NTT_int64_U_arr, iters, (height, width))
    print ('psnr_U_arr = ', psnr_U_arr)

# cost V
#     ccc_cost_V = []
#     for k in range(iters):
#         temp = Ori_int64_V_arr[k] - NTT_int64_V_arr[k]
#         temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#         temp2 = np.array(temp2)
#         temp3 = np.sum(temp2)
#         ccc = (1.0 / (width * height)) * temp3
#         ccc_cost_V.append(ccc)
#         print ('calculate {}th frm -- U'.format(k))
#
#     print (ccc_cost_V )  # MSE 矩阵 <type 'list'>
#     ccc_cost_V_arr = np.array(ccc_cost_V)
#     print (ccc_cost_V_arr ) # <type 'numpy.ndarray'>
#     print (ccc_cost_V_arr.shape )   # (40,)
#
#     ccc_psnr_V_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_V_arr))   # psnr 矩阵
#     print ('ccc_psnr_V_arr = ', ccc_psnr_V_arr)

    psnr_V_arr = cal_psnr(Ori_int64_V_arr, NTT_int64_V_arr, iters, (height, width))
    print ('psnr_V_arr = ', psnr_V_arr)


    time4 = time.clock()

# 输出 csv 文件
    time5 = time.clock()
    test_Y = pd.Series(psnr_Y_arr)
    test_U = pd.Series(psnr_U_arr)
    test_V = pd.Series(psnr_V_arr)
    test_none = pd.Series([])
    test_SSIM = pd.Series(SSIM)
    test = pd.DataFrame({'Y_psnr': test_Y, 'U_psnr': test_U, 'V_psnr': test_V, '': test_none, 'SSIM':test_SSIM})
    test = test[['Y_psnr', 'U_psnr', 'V_psnr', '', 'SSIM']]     # 按照YUV的顺序进行输出
    test.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test.csv')
    print ('start show...')

    cv2.waitKey(0)
end = time.clock()
print('Running time: %s Seconds' % (end - start))

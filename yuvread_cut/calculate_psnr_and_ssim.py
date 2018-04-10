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
    NTT_Y = []
    NTT_U = []
    NTT_V = []
    for i in range(iters):
        data_NTT = yuv_import(url_NTT, (height, width), 1, i)
        NTT_Y.append(data_NTT[0][0])
        NTT_U.append(data_NTT[1][0])
        NTT_V.append(data_NTT[2][0])
        # cv2.imshow("sohow{}".format(i), YYx[i])
        print 'This is {}th frm -- NTT'.format(i)

    NTT_Y_arr = np.array(NTT_Y)  # YYx_Y_arr： NTT图像数据， uint8 类型
    NTT_U_arr = np.array(NTT_U)
    NTT_V_arr = np.array(NTT_V)
    print 'The shape of NTT_Y_arr.shape is :', NTT_Y_arr.shape   # (2, 1080, 1920) 同样2表示两帧视频

    # print NTT_Y_arr.shape

    NTT_int8_Y = []
    NTT_int8_U = []
    NTT_int8_V = []
    for i in range(iters):
        arr2_Y = NTT_Y_arr[i].astype(int8)
        NTT_int8_Y.append(arr2_Y)
        arr2_U = NTT_U_arr[i].astype(int8)
        NTT_int8_U.append(arr2_U)
        arr2_V = NTT_V_arr[i].astype(int8)
        NTT_int8_V.append(arr2_V)
        print 'change NTT {}th frm'.format(i)

    NTT_int8_Y_arr = np.array(NTT_int8_Y)  # NTT int8类型数据
    NTT_int8_U_arr = np.array(NTT_int8_U)  # NTT int8类型数据
    NTT_int8_V_arr = np.array(NTT_int8_V)  # NTT int8类型数据
    print NTT_int8_Y_arr.shape  # (10, 1080, 1920)

# SSIM
#     SSIM  用 uint8 类型的数据 ,     x 表示 Ori , y 表示 NTT

    # 计算每一帧的均值ux
    ux_Y = []
    uy_Y = []
    for i in range(iters):
        temp_Ori = np.mean(Ori_Y_arr[i])    # 第i帧的均值
        ux_Y.append(temp_Ori)
        temp_NTT = np.mean(NTT_Y_arr[i])
        uy_Y.append(temp_NTT)

    # print type(ux_Y)    # <type 'list'>
    ux_Y_arr = np.array(ux_Y)
    uy_Y_arr = np.array(uy_Y)
    # print ux_Y_arr  # [106.14581983 106.30893615 106.2186304 ]
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
    meanx_arr = np.ones((iters, height, width))
    for k in range(iters):
        for i in range(len(meanx_arr[0])):
            for j in range(len(meanx_arr[0][0])):
                # if meanx_arr[k][i][j] == 1:
                    # print 'ux_Y_arr[{}] is {}'.format(k, ux_Y_arr[k])
                meanx_arr[k][i][j] = ux_Y_arr[k]

    # print ux_Y_arr
    # print meanx_arr         # 每一帧存放该帧的均值
    # print meanx_arr.shape   # (3, 1080, 1920)
    sigma2x_np = []
    for i in range(iters):
        temp1 = Ori_Y_arr[i] - meanx_arr[i]
        print temp1.shape
        temp2 = [[temp1[i][j] ** 2 for j in range(len(temp1[i]))] for i in range(len(temp1))]
        # temp2 = temp1 ** 2
        temp3 = np.sum(temp2)
        temp4 = (1.0 / ((height * width) - 1)) * temp3
        sigma2x_np.append(temp4)

    print type(sigma2x_np)
    sigma2x_np = np.array(sigma2x_np)       # (3,)
    sigma2x_np = sigma2x_np[:, np.newaxis]  # (3, 1)    Ori 方差
    print 'sigma2x_np.shape', sigma2x_np.shape
    print 'sigma2x_np', sigma2x_np
    time2_end = time.clock()
    print 'numpy矩阵操作用时:{} seconds...'.format(time2_end - time2)

    sigma_x_np = sqrt(sigma2x_np)   # Ori 标准差
    print (sigma_x_np)

    # NTT的每一帧方差
    meany_arr = np.ones((iters, height, width))
    for k in range(iters):
        for i in range(len(meany_arr[0])):
            for j in range(len(meany_arr[0][0])):
                meany_arr[k][i][j] = uy_Y_arr[k]

    sigma2y_np = []
    for i in range(iters):
        temp1 = NTT_Y_arr[i] - meany_arr[i]
        print temp1.shape
        temp2 = [[temp1[i][j] ** 2 for j in range(len(temp1[i]))] for i in range(len(temp1))]
        temp3 = np.sum(temp2)
        temp4 = (1.0 / (height * width)) * temp3
        sigma2y_np.append(temp4)

    sigma2y_np = np.array(sigma2y_np)       # (3,)
    sigma2y_np = sigma2y_np[:, np.newaxis]  # (3, 1)    NTT 方差
    sigma_y_np = sqrt(sigma2y_np)       # NTT 标准差
    print (sigma_y_np)

    # Ori 和 NTT 对应两帧的协方差
    ux_Y_arr_int8 = ux_Y_arr.astype(float)
    print ux_Y_arr_int8.shape  # (3, 1)
    uy_Y_arr_int8 = uy_Y_arr.astype(float)
    # Ori_int8_Y_arr   (2, 1080, 1920) 同样2表示两帧视频
    temp_arr = []
    for k in range(iters):
        for i in range(height):
            for j in range(width):
                temp1 = Ori_int8_Y_arr[k][i][j] - ux_Y_arr_int8[k]
                temp2 = NTT_int8_Y_arr[k][i][j] - ux_Y_arr_int8[k]
                temp3 = temp1 * temp2
                temp_arr.append(temp3)
        print 'calculate {}th frm of pre-conv'.format(k)
    temp_arr = np.array(temp_arr)       # 计算 (X(i,j) - ux)*(Y(i,j) - uy)得到的矩阵
    print temp_arr.shape
    temp_arr = reshape(temp_arr, (iters, height, width))
    print temp_arr.shape
    print temp_arr[0].shape
    print '****************************************'
    conv_sigma_xy = []
    for i in range(iters):
        temp_arr_1frm_sum = np.sum(temp_arr[i])     # 第i帧的(X(i,j) - ux)*(Y(i,j) - uy)求和
        print type(temp_arr_1frm_sum)       # <type 'numpy.float64'>
        print temp_arr_1frm_sum
        print temp_arr_1frm_sum.shape   # ()

        temp_conv_sigma_xy = (1.0 / ((width * height) - 1)) * temp_arr_1frm_sum
        print temp_conv_sigma_xy    # 9582.09975141
        conv_sigma_xy.append(temp_conv_sigma_xy)
    conv_sigma_xy_arr = np.array(conv_sigma_xy)
    conv_sigma_xy_arr = conv_sigma_xy_arr[:, np.newaxis]
    print conv_sigma_xy_arr.shape       # (3, 1)
    print conv_sigma_xy_arr         # Ori NTT 对应帧的协方差
    # print temp_arr[0]
    # print temp_arr.shape
    # a = np.sum(temp_arr[0])
    # b = (1.0 / (height * width)) * a
    # print a
    # print b
    K1 = 0.01
    K2 = 0.03
    LL = 255
    C1 = (K1 * LL) ** 2
    C2 = (K2 * LL) ** 2
    C3 = C2 / 2
    # L(x,y)
    L = []
    for i in range(iters):
        L_temp = (2 * ux_Y_arr[i] * uy_Y_arr[i] + C1) / (sigma2x_np[i] + sigma2y_np[i] + C1)
        L.append(L_temp)
    L = np.array(L)
    print L

    # C(x,y)
    C = []
    for i in range(iters):
        C_temp = (2 * sigma_x_np[i] * sigma_y_np[i] + C2)/(sigma2x_np[i] + sigma2y_np[i] + C2)
        C.append(C_temp)
    C = np.array(C)
    print C

    # S(x,y)
    S = []
    for i in range(iters):
        S_temp = (conv_sigma_xy_arr[i] + C3)/(sigma_x_np[i] * sigma_y_np[i] + C3)
        S.append(S_temp)
    S = np.array(S)
    print S

    # SSIM score
    SSIM = []
    for i in range(iters):
        SSIM_temp = L[i] * C[i] * S[i]
        SSIM.append(SSIM_temp)
    SSIM = np.array(SSIM)
    print SSIM







    # print ux_Y_arr.shape  # (3,1)
    # print ux_Y_arr[0]   # [106.14581983]
    # print ux_Y_arr[1]   # [106.30893615]
    # print ux_Y_arr[2]   # [106.2186304]

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

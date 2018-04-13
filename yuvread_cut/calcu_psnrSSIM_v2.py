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

    # 尝试 numpy的矩阵操作
    # Ori的每一帧方差
    meanx_arr = np.ones((iters, height, width))  # 构造所有元素都等于均值的矩阵
    for k in range(iters):
        for i in range(len(meanx_arr[0])):  # 1080次
            for j in range(len(meanx_arr[0][0])):  # 1920次
                # if meanx_arr[k][i][j] == 1:
                # print 'ux_Y_arr[{}] is {}'.format(k, ux_Y_arr[k])
                meanx_arr[k][i][j] = ux_Y_arr[k]        # meanx_arr[0] # 每一帧存放该帧的均值 meanx_arr.shape   # (3, 1080, 1920)
        print('calculate {}th frm -- meanx_arr'.format(k))
    print('**************1111111111111111111111111111111111111111111111111111111111111111**********')

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
    # print('sigma2x_np.shape', sigma2x_np.shape)
    # print('sigma2x_np', sigma2x_np)

    sigma_x_np = sqrt(sigma2x_np)  # Ori 标准差  就一列数据,每行代表一帧的 标准差
    # print(sigma_x_np)

    # NTT的每一帧方差
    meany_arr = np.ones((iters, height, width))
    for k in range(iters):
        for i in range(len(meany_arr[0])):
            for j in range(len(meany_arr[0][0])):
                meany_arr[k][i][j] = uy_Y_arr[k]
        print ('calculate {}th frm -- meany_arr'.format(k))

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
    # print (sigma_y_np)

    multi_xy = []  # 存放计算 (X(i,j) - ux)*(Y(i,j) - uy)得到的矩阵
    for i in range(iters):
        mult_temp = (Ori_int64_Y_arr[i] - meanx_arr[i]) * (NTT_int64_Y_arr[i] - meany_arr[i])
        multi_xy.append(mult_temp)
        print ('calculate {}th frm -- multi_xy'.format(i))
    multi_xy = np.array(multi_xy)  # multi_xy.shape   (2, 1, 1080, 1920)

    print ('****************************************')
    conv_sigma_xy = []
    for i in range(iters):
        temp_arr_1frm_sum = np.sum(multi_xy[i])  # 第i帧的(X(i,j) - ux)*(Y(i,j) - uy)求和
        temp_conv_sigma_xy = (1.0 / ((width * height) - 1)) * temp_arr_1frm_sum  # 协方差计算公式
        conv_sigma_xy.append(temp_conv_sigma_xy)
        print ('calculate {}th frm of conv_sigma_xy'.format(i))

    conv_sigma_xy_arr = np.array(conv_sigma_xy)
    conv_sigma_xy_arr = conv_sigma_xy_arr[:, np.newaxis]  # (3, 1)   多行一列,每行存放协方差
    # print (conv_sigma_xy_arr)  # Ori NTT 对应帧的协方差

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
        psnr_list.append(ccc)       # print (psnr_list ) # MSE 矩阵 <type 'list'>
        print ('calculate {}th frm -- Y'.format(k))

    psnr_list_arr = np.array(psnr_list)     # <type 'numpy.ndarray'> ; (psnr_list_arr.shape )  (40,)

    psnr_Y_arr = 20 * np.log10(255 / np.sqrt(psnr_list_arr))  # psnr 矩阵
    print ('psnr_Y_arr = ', psnr_Y_arr)
    return psnr_Y_arr


if __name__ == '__main__':
    width = 1920
    height = 1080

    url_ori = '/home/lx/Videos/CrowdRun_1080p50.yuv'
    url_NTT = '/home/lx/Videos/NTT_1080p50_10Mbps_8bit_500of1010.yuv'

    iters = 4
    piece = 2

    time0 = time.clock()
    k = 0
    iters_temp = 0
    if iters > piece:

        for i in range(iters_temp, iters_temp + piece):
            print ('This is {}th calculation'.format(i))
            # 读取ori的 iters 帧图像
            # Ori_int64_YUV_arr = yuv_readfrms_int64(url_ori, (height, width), i)
            Ori_int64_YUV_arr = yuv_import(url_ori, (height, width), 1, i)

            Ori_int64_Y_arr = Ori_int64_YUV_arr[0]      # Ori 图像Y分量数据 int64类型
            Ori_int64_U_arr = Ori_int64_YUV_arr[1]      # Ori 图像U分量数据 int64类型
            Ori_int64_V_arr = Ori_int64_YUV_arr[2]      # Ori 图像V分量数据 int64类型

            # 读取NTT的 iters 帧图像
            # NTT_int64_YUV_arr = yuv_readfrms_int64(url_NTT, (height, width), i)
            NTT_int64_YUV_arr = yuv_import(url_NTT, (height, width), i)

            NTT_int64_Y_arr = NTT_int64_YUV_arr[0]
            NTT_int64_U_arr = NTT_int64_YUV_arr[1]
            NTT_int64_V_arr = NTT_int64_YUV_arr[2]

            # calculate SSIM******************************************************************************************
            SSIM = cal_ssim(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width))

            # calculate PSNR*******************************************************************************************
            # #cost Y
            psnr_Y_arr = cal_psnr(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width))
            # print ('psnr_Y_arr = ', psnr_Y_arr)
            # cost U
            psnr_U_arr = cal_psnr(Ori_int64_U_arr, NTT_int64_U_arr, iters, (height, width))
            # print ('psnr_U_arr = ', psnr_U_arr)
            # cost V
            psnr_V_arr = cal_psnr(Ori_int64_V_arr, NTT_int64_V_arr, iters, (height, width))
            # print ('psnr_V_arr = ', psnr_V_arr)

            test_Y = pd.Series(psnr_Y_arr)
            test_U = pd.Series(psnr_U_arr)
            test_V = pd.Series(psnr_V_arr)
            test_none = pd.Series([])
            test_SSIM = pd.Series(SSIM)
            test = pd.DataFrame({'Y_psnr': test_Y, 'U_psnr': test_U, 'V_psnr': test_V, '': test_none, 'SSIM':test_SSIM})

            name = ['Y_psnr', 'U_psnr', 'V_psnr', '', 'SSIM']
            if iters_temp == 0:
                df = pd.DataFrame([test_Y, test_U, test_V, test_none, test_SSIM], columns=name)
                df.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test_HHH.csv')
            else:
                df2 = pd.DataFrame([test_Y, test_U, test_V, test_none, test_SSIM], columns=name)
                df = df.append(df2, ignore_index=True)
                df.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test_HHH.csv')
        iters_temp = iters_temp + piece
        iters = iters - piece
    # else:   # 输出数据



            # test = test[['Y_psnr', 'U_psnr', 'V_psnr', '', 'SSIM']]     # 按照YUV的顺序进行输出
            # test.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test.csv')

    time1 = time.clock()

# # 读取ori的 iters 帧图像
#     Ori_int64_YUV_arr = yuv_readfrms_int64(url_ori, (height, width), iters)
#     Ori_int64_Y_arr = Ori_int64_YUV_arr[0]      # Ori 图像Y分量数据 int64类型
#     Ori_int64_U_arr = Ori_int64_YUV_arr[1]      # Ori 图像U分量数据 int64类型
#     Ori_int64_V_arr = Ori_int64_YUV_arr[2]      # Ori 图像V分量数据 int64类型
#
# # 读取NTT的 iters 帧图像
#     NTT_int64_YUV_arr = yuv_readfrms_int64(url_NTT, (height, width), iters)
#     NTT_int64_Y_arr = NTT_int64_YUV_arr[0]
#     NTT_int64_U_arr = NTT_int64_YUV_arr[1]
#     NTT_int64_V_arr = NTT_int64_YUV_arr[2]
#
# # calculate SSIM*******************************************************************************************************
#     SSIM = cal_ssim(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width))
#
# # calculate PSNR*******************************************************************************************************
# # #cost Y
#     psnr_Y_arr = cal_psnr(Ori_int64_Y_arr, NTT_int64_Y_arr, iters, (height, width))
#     # print ('psnr_Y_arr = ', psnr_Y_arr)
# # cost U
#     psnr_U_arr = cal_psnr(Ori_int64_U_arr, NTT_int64_U_arr, iters, (height, width))
#     # print ('psnr_U_arr = ', psnr_U_arr)
# # cost V
#     psnr_V_arr = cal_psnr(Ori_int64_V_arr, NTT_int64_V_arr, iters, (height, width))
#     # print ('psnr_V_arr = ', psnr_V_arr)
#
#     time2 = time.clock()
# # 输出 csv 文件
#     time5 = time.clock()
#     test_Y = pd.Series(psnr_Y_arr)
#     test_U = pd.Series(psnr_U_arr)
#     test_V = pd.Series(psnr_V_arr)
#     test_none = pd.Series([])
#     test_SSIM = pd.Series(SSIM)
#     test = pd.DataFrame({'Y_psnr': test_Y, 'U_psnr': test_U, 'V_psnr': test_V, '': test_none, 'SSIM':test_SSIM})
#     test = test[['Y_psnr', 'U_psnr', 'V_psnr', '', 'SSIM']]     # 按照YUV的顺序进行输出
#     test.to_csv('/home/lx/Videos/new_ver_NTT_and_SSIM_test.csv')
#     print ('start show...')

    time3 = time.clock()

    cv2.waitKey(0)
end = time.clock()
print('Running time: %s Seconds' % (end - start))

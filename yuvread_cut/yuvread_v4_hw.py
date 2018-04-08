#coding:utf-8
#!/usr/bin/env python

# test git update
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
    print dims[0]
    print dims[1]
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    print d00
    print d01
    Yt = zeros((dims[0], dims[1]), uint8, 'C')
    Ut = zeros((d00, d01), uint8, 'C')
    Vt = zeros((d00, d01), uint8, 'C')

    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                print m, n
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

    url_ori = '/home/lx/Videos/CrowdRun_1080p50-0_10frms.yuv'
    # url_NTT = '/home/lx/Videos/NTT_10frms.yuv'
    url_NTT = '/home/lx/Videos/NTT_50frms_cut40frms.yuv'

    iters = 40
    datas_ori = []
    datas_NTT = []
    costs = []

#
# #获取源的第一帧
#     for i in range(iters):
#         data = yuv_import(url_ori, (height, width), iters, i)
#         datas_ori.append(data[0][i])
#         cv2.imshow("sohow{}".format(i), data[0][i])
# #
# #     a = np.array(datas_ori)
# #     print a.shape


    #
    # data_ori = yuv_import(url_ori, (height, width), 1, 0)  # 源视频就读取第一帧   注意：第三个位置要写iters，写１报错，原因不详
    # a = np.array(data_ori)
    #
    # data_ori2 = yuv_import(url_ori, (height, width), 1, 1)  # 源视频就读取第一帧
    # a2 = np.array(data_ori2)
    # print a.shape  # (3, 3)
    # print a2.shape  # (3, 3)
    #
    # # # 查看第六帧画面
    # YY = data_ori[0][0]
    # cv2.imshow("sohow", YY)
    #
    # YYa2 = data_ori2[0][0]
    # cv2.imshow("sohow2", YYa2)
    #
    # kk = np.sum(data_ori[0][0] - data_ori2[0][0])
    # print kk
    #
    # # 6th frm
    # data_ori6 = yuv_import(url_ori, (height, width), 1, 5)  # 源视频就读取第一帧
    # YYa6 = data_ori6[0][0]
    # cv2.imshow("sohow6", YYa6)
    #
    # # 8th frm
    # data_ori8 = yuv_import(url_ori, (height, width), 1, 7)  # 源视频就读取第一帧
    # YYa8 = data_ori8[0][0]
    # cv2.imshow("sohow8", YYa8)
    #
    # # 10th frm
    # data_ori10 = yuv_import(url_ori, (height, width), 1, 9)  # 源视频就读取第一帧
    # YYa10 = data_ori10[0][0]
    # cv2.imshow("sohow10", YYa10)
    #
# 读取ori的第一帧图像
    data_ori_1st_frm = yuv_import(url_ori, (height, width), 1, 0)
    # XX_arr = np.array(data_ori_1st_frm)
    arr = data_ori_1st_frm[0][0].astype(int8)



# 一次性读取NTT十帧图像
    YYx = []
    for i in range(iters):
        data_NTT_10frms = yuv_import(url_NTT, (height, width), 1, i)
        YYx.append(data_NTT_10frms[0][0])
        # cv2.imshow("sohow{}".format(i), YYx[i])

    YYx_arr = np.array(YYx)
    print YYx_arr.shape     # (10, 1080, 1920)
    print arr.shape

    arr2s = []
    for i in range(iters):
        arr2 = YYx_arr[i].astype(int8)
        arr2s.append(arr2)

    arr2_arr = np.array(arr2s)  # NTT int8类型数据
    print arr2_arr.shape    # (10, 1080, 1920)

    ccc_cost = []
    for k in range(iters):
       temp = arr - arr2_arr[k]
       temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
       temp2 = np.array(temp2)
       temp3 = np.sum(temp2)
       ccc = (1.0 / (width * height)) * temp3
       ccc_cost.append(ccc)

    print ccc_cost   # MSE 矩阵 <type 'list'>
    ccc_cost_arr = np.array(ccc_cost)
    print ccc_cost_arr  # <type 'numpy.ndarray'>
    print ccc_cost_arr.shape    # (40,)

    ccc_psnr_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_arr))   # psnr 矩阵
    print 'ccc_psnr_arr = ', ccc_psnr_arr

# 输出 csv 文件
    name = ['Y_psnr']
    test = pd.DataFrame(columns=name, data=ccc_psnr_arr)
    # test.to_csv('/home/d066/Videos/NTT.csv')
    test.to_csv('/home/lx/Videos/NTT.csv')


    #     for i in range(iters):
    #         data = yuv_import(url_NTT, (height, width), iters, i)
    #         datas_NTT.append(data[0][i])
    #
    #     c = np.array(datas_NTT)
    #     arr2s = []
    #     for i in range(iters):
    #         arr2 = c[i].astype(int8)
    #         arr2s.append(arr2)
    #
    #     arr2_arr = np.array(arr2s)
    #
    #


    # YY1 = data_ori[0][1]
    # cv2.imshow("sohow1", YY1)
    # # print data_ori[0][0] == data_ori[0][1]
    #
    # YY2 = data_ori[0][2]
    # cv2.imshow("sohow2", YY2)
    # # print data_ori[0][0] == data_ori[0][2]
    # kk = np.sum(data_ori[0][0] -  data_ori[0][1])
    # print kk


    #
    # data_ori5 = yuv_import(url_ori, (height, width), iters, 5)  # 源视频就读取第一帧   注意：第三个位置要写iters，写１报错，原因不详
    # a5 = np.array(data_ori5)
    # print a5.shape  # (3, 3)
    #
    # YY5 = data_ori5[0][5]
    # cv2.imshow("sohow5", YY5)


#     # test **********************
#     print a[0][0].shape
#     arr = a[0][0].astype(int8)
#
#     for i in range(iters):
#         data = yuv_import(url_NTT, (height, width), iters, i)
#         datas_NTT.append(data[0][i])
#
#     c = np.array(datas_NTT)
#     arr2s = []
#     for i in range(iters):
#         arr2 = c[i].astype(int8)
#         arr2s.append(arr2)
#
#     arr2_arr = np.array(arr2s)
#
#
#     print a.shape  # (3, 3)
#     print a2.shape  # (3, 3)
#
#     print a[0][0] == a2[0][0]
#
#     ccc_cost = []
#     for k in range(iters):
#        temp = arr - arr2_arr[k]
#        temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#        temp2 = np.array(temp2)
#        temp3 = np.sum(temp2)
#        ccc = (1.0 / (width * height)) * temp3
#        ccc_cost.append(ccc)
#
#     print ccc_cost   # MSE 矩阵 <type 'list'>
#     ccc_cost_arr = np.array(ccc_cost)
#     print ccc_cost_arr  # <type 'numpy.ndarray'>
#
#     ccc_psnr_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_arr))   # psnr 矩阵
#     print 'ccc_psnr_arr = ', ccc_psnr_arr
# # test ********************************************************************8


#
#
# # 获取NTT的视频帧
#     for i in range(iters):
#         data = yuv_import(url_NTT, (height, width), iters, i)
#         datas_NTT.append(data[0][i])
#         # cv2.imshow("sohow{}".format(i), data[0][i])
#iters
#     # for i in range(iters):
#     #     cv2.imshow("sohow{}".format(i), datas_NTT[i])
#
#     # print type(datas_NTT)   # <type 'list'>
#     c = np.array(datas_NTT)
#     # print c.shape    # (3, 1080, 1920)
#
#     arr = a[0][0].astype(int8)
#
#     arr2s = []
#     for i in range(iters):
#         arr2 = c[i].astype(int8)
#         arr2s.append(arr2)
#
#     arr2_arr = np.array(arr2s)
#
#     ccc_cost = []
#     for k in range(iters):
#         temp = arr - arr2_arr[k]
#         temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
#         temp2 = np.array(temp2)
#         temp3 = np.sum(temp2)
#         ccc = (1.0 / (width * height)) * temp3
#         ccc_cost.append(ccc)
#
#     print ccc_cost   # MSE 矩阵 <type 'list'>
#     ccc_cost_arr = np.array(ccc_cost)
#     print ccc_cost_arr  # <type 'numpy.ndarray'>
#
#     ccc_psnr_arr = 20 * np.log10(255 / np.sqrt(ccc_cost_arr))   # psnr 矩阵
#     print 'ccc_psnr_arr = ', ccc_psnr_arr
#
#     name = ['Y_psnr']
#     test = pd.DataFrame(columns=name, data=ccc_psnr_arr)
#     # test.to_csv('/home/d066/Videos/NTT.csv')
#     test.to_csv('/home/lx/Videos/NTT.csv')
#



    print 'start show...'


    # for i in range(len(data)):
    #     print i
    # cv2.imshow("sohow", YY)
    # cv2.imshow("sohow2", YY2)
    # cv2.imshow("sohow22", YY22)

    cv2.waitKey(0)

print '???????/'
end = time.clock()
print('Running time: %s Seconds' % (end-start))

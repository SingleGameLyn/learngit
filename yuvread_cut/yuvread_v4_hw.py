#coding:utf-8
#!/usr/bin/env python

# time 2018.4.9
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

    url_ori = '/home/lx/Videos/CrowdRun_1080p50-0_10frms.yuv'
    # url_NTT = '/home/lx/Videos/NTT_10frms.yuv'
    url_NTT = '/home/lx/Videos/NTT_1080p50_10Mbps_8bit.yuv'

    iters = 10
    datas_ori = []
    datas_NTT = []
    costs = []

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
        print 'This is {}th frm -- NTT'.format(i)

    YYx_arr = np.array(YYx)
    print YYx_arr.shape     # (10, 1080, 1920)
    print arr.shape

    arr2s = []
    for i in range(iters):
        arr2 = YYx_arr[i].astype(int8)
        arr2s.append(arr2)
        print 'change NTT {}th frm -- NTT'.format(i)

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
        print 'calculate {}th frm'.format(k)

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

    print 'start show...'


    # for i in range(len(data)):
    #     print i
    # cv2.imshow("sohow", YY)
    # cv2.imshow("sohow2", YY2)
    # cv2.imshow("sohow22", YY22)

    cv2.waitKey(0)

end = time.clock()
print('Running time: %s Seconds' % (end-start))

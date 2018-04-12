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
    return (Y, U, V)


if __name__ == '__main__':
    width = 1920
    height = 1080

    url_ori = '/home/lx/Videos/CrowdRun_1080p50.yuv'
    url_NTT = '/home/lx/Videos/NTT_50frms_cut40frms.yuv'

    iters = 20

# 读取ori的第一帧图像
    data_ori_1st_frm = yuv_import(url_ori, (height, width), 1, 1)   # ori第一帧图像的数据,uint8类型   <type 'tuple'>
    Ori_int64_list = ndarray.tolist(data_ori_1st_frm[0][0])   # 找psnr的峰值,只需要用Y分量即可, 先转成list   <type 'list'>
    Ori_int64_arr = array(Ori_int64_list)   # 在转成 array , 得到 int64类型数据   <type 'numpy.ndarray'>

# 一次性读取NTT十帧图像
    NTT = []
    for i in range(iters):
        data_NTT_10frms = yuv_import(url_NTT, (height, width), 1, i)
        NTT.append(data_NTT_10frms[0][0])
        # cv2.imshow("sohow{}".format(i), NTT[i])
        print 'This is {}th frm -- NTT'.format(i)

    NTT_arr = np.array(NTT)     # NTT uint8类型数据    (10, 1080, 1920)

    NTT_list = ndarray.tolist(NTT_arr)  # 先转list
    NTT_int64_arr = np.array(NTT_list)      # 再转array, 得到 NTT int8类型数据   (10, 1080, 1920)

    mse = []
    for k in range(iters):
        temp = Ori_int64_arr - NTT_int64_arr[k]
        temp2 = [[temp[i][j] ** 2 for j in range(len(temp[i]))] for i in range(len(temp))]
        temp2 = np.array(temp2)
        temp3 = np.sum(temp2)
        ccc = (1.0 / (width * height)) * temp3
        mse.append(ccc)
        print 'calculate {}th frm'.format(k)

    # print mse   # MSE 矩阵 <type 'list'>
    mse_arr = np.array(mse)       # <type 'numpy.ndarray'>    shape:(40,)

    psnr_arr = 20 * np.log10(255 / np.sqrt(mse_arr))   # psnr 矩阵
    print 'psnr_arr = ', psnr_arr

# 输出 csv 文件
    name = ['Y_psnr']
    psnr_data = pd.DataFrame(columns=name, data=psnr_arr)
    # psnr_data.to_csv('/home/d066/Videos/NTT.csv')
    psnr_data.to_csv('/home/lx/Videos/NTT_ppt.csv')

    print 'start show...'

    cv2.waitKey(0)

end = time.clock()
print('Running time: %s Seconds' % (end-start))

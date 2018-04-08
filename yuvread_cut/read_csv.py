import csv
import numpy as np
import os

# csvFile = open('/home/d066/Videos/NTT.csv', "r")
# reader = csv.reader(csvFile)
# print type(reader)
# psnr_arr = np.array(reader)
# print type(psnr_arr)
# print psnr_arr.shape
# # print len(reader)
# for item in reader:
#     print(item)
#
# psnr_ar = []
# for i in reader:
#     psnr_ar.append(i)
#
# print 'psnr_ar = ', psnr_ar

import csv
with open('/home/lx/Videos/NTT.csv', 'U') as csvfile:
    reader = csv.DictReader(csvfile)
    column = [row['Y_psnr'] for row in reader]
# print column

print type(column)  # <type 'list'>
# print column.max()

psnr_ar = np.array(column)
print psnr_ar.shape      # (5,)
# print psnr_ar
psnr_arr = psnr_ar[:, np.newaxis]
print type(psnr_arr)
print psnr_arr.shape       # (5, 1)
# print psnr_arr

# for i in range(len(psnr_arr)):
#     print '{}th row is {}'.format(i, psnr_arr[i])


print max(psnr_arr)

print max(column)
print column.index(max(column))

height = 1080
width = 1920
length = 10
blk_size = (height * width) * 3 / 2
start = column.index(max(column)) * blk_size
end = start + length * blk_size
cut_ori = '/home/lx/Videos/NTT_repeat_40frms.yuv'

os.system("del_data_yuv_start_end.exe   %s   %s  %s" % (cut_ori, start, end))

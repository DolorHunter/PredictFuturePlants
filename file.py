# coding=utf-8

import numpy as np
from PIL import Image

ROW_SIZE = 12  # 36 [test size] 12
COL_SIZE = 12  # 36 [test size] 12
TIME_SIZE = 10000
Z1_LOC = '.\\data\\Z1\\Z1-'
Z1_IMAGE = 200          # 前200张图作为训练集
Z1_TEST_IMAGE = 12      # 后12张图作为测试集


# Convert image to array[1*1296] in float32
def arr_image_z1(num):
    if num >= 1000:
        print("Too many image! Maximum scale is 999.")
    else:
        units = int(num % 10)
        tens = int((num / 10) % 10)
        hundreds = int((num / 100) % 10)
        str_num = chr(hundreds + 48) + chr(tens + 48) + chr(units + 48)
        im = Image.open(Z1_LOC + str_num + '.tif')
        im = im.resize((ROW_SIZE, COL_SIZE), Image.ANTIALIAS)
        im_arr = np.array(im.convert('L'))  # Convert to array
        nm_arr = im_arr.reshape([1, ROW_SIZE*COL_SIZE])
        nm_arr = nm_arr.astype(np.float32)
        arr_ready = np.multiply(nm_arr, 1/255)
        return arr_ready


def label_image_z1(num):
    matrix = np.zeros(TIME_SIZE)
    matrix[num + 1] = 1
    matrix = matrix.reshape([1, TIME_SIZE])
    matrix.astype(np.float32)
    return matrix


# TEST
def input_data_z1():
    for i in range(1, Z1_IMAGE+1):
        arr_image_z1(i)
        label_image_z1(i)


def main():
    input_data_z1()
    return 0


if __name__ == '__main__':
    main()

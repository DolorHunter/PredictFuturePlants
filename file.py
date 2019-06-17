# coding=utf-8

import numpy as np
from PIL import Image

ROW_SIZE = 1200
COL_SIZE = 1200
TIME_SIZE = 10000
Z1_LOC = '.\\data\\Z1\\Z1-'
Z1_IMAGE = 212


# Convert image to array[1*1440000] in float32
def arr_image_z1(num):
    if num >= 1000:
        print("Too many image! Maximum scale is 999.")
    else:
        units = int(num % 10)
        tens = int((num / 10) % 10)
        hundreds = int((num / 100) % 10)
        str_num = chr(hundreds + 48) + chr(tens + 48) + chr(units + 48)
        im = Image.open(Z1_LOC + str_num + '.tif')
        # im = im.resize((28, 28), Image.ANTIALIAS)
        im_arr = np.array(im.convert('L'))  # Convert to array
        nm_arr = im_arr.reshape([1, 1440000])
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

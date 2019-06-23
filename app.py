# coding=utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import forward
import backward
import file


Z1_PREDICT_PATH = '.\\predict\\Z1\\Z1-'

def restore_model(time):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, 0)

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                pre_array = sess.run(y, feed_dict={x: time})
                return pre_array
            else:
                print("No checkpoint file found")
                return -1


# 预测矩阵转图片保存
def image_arr(time):
    matrix = np.zeros(file.TIME_SIZE)
    matrix[time + 1] = 1
    matrix = matrix.reshape([1, file.TIME_SIZE])
    matrix.astype(np.float32)
    arr = restore_model(matrix)
    arr = arr.reshape((file.ROW_SIZE, file.COL_SIZE))
    arr = np.multiply(arr, 255)
    new_dimension = (1200, 1200)
    arr_ready = cv2.resize(arr, new_dimension, interpolation=cv2.INTER_LANCZOS4)  # 8x8像素邻域的Lanczos插值
    im = Image.fromarray(arr_ready)  # 展示当前图片
    plt.show(im)
    im.save(Z1_PREDICT_PATH + str(time) + ".tif")


def main():
    time = 230  # int(input("Input the time:"))
    image_arr(time)


if __name__ == '__main__':
    main()

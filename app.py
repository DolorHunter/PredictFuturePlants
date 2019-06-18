# coding=utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import forward
import backward
import file


def restore_model(time):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
		y = forward.forward(x, None)
		# pre_value = tf.argmax(y, 1)

		variable_averages = tf.trainExponentialMovingAverage(backward.MOVING_AVERAGE_DEcAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

				pre_value = sess.run(y, feed_dict={x: time})
				return pre_value
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
	arr = arr.reshape([1200*1200])
	arr_ready = np.multiply(arr, 255)
	im = Image.fromarray(arr_ready)
	im.save(file.Z1_LOC + time + ".tif")


def main():
	time = int(input("Input the time:"))
	image_arr(time)


if __name__ == '__main__':
	main()

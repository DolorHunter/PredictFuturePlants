# coding=utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import forward
import backward
import file


def restore_model(matx):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
		print("----test1----")
		result = forward.forward(x, 0.99)     # y = forward.forward(x, None)   /  y改成了result
		# print(type(result))            在此处显示result是一个  numpy.ndarray

		variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
	# backward.MOVING_AVERAGE_DEcAY / tf.trainExponentialMovingAverage
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				# resultR = tf.convert_to_tensor(result.tostring())
				print('result:', result)
				# print('resultR:', resultR)
				pre_array = sess.run(result, feed_dict={x: matx})
				print(type(pre_array))
				print("pass")
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
	# matrix = tf.convert_to_tensor(matrix)     提示错误:The value of a feed cannot be a tf.Tensor object.说明转换成功
	arr = restore_model(matrix)
	print(type(arr))
	print(arr)
	# tf.constant(arr, dtype=tf.float32)

	'''
	Farr = tf.to_float(arr, name="ToFloat")
	print(arr)
	print(Farr)
	print(type(Farr))
	'''
	arr.astype(np.float32)
	# arr = arr.reshape([1200*1200])                                   # arr = arr.resize([1200*1200])

	arr_ready = np.multiply(arr, 255)
	im = Image.fromarray(arr_ready)
	im.save(file.Z1_LOC + time + ".tif")


def main():
	time = int(input("Input the time:"))
	image_arr(time)


if __name__ == '__main__':
	main()

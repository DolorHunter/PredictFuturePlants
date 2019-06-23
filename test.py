# coding=utf-8

import time
import tensorflow as tf
import forward
import backward
import file

TEST_INTERVAL_SECS = 60


# 用于实时监测准确度的测试程序
def test():
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
		y_= tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
		y = forward.forward(x, 0)

		ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		correct_prediction = tf.greater_equal(y, y_)  # 预测值大于等于真实值即为预测正确
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

					for i in range(file.Z1_TEST_IMAGE):
						xs_test_label = file.label_image_z1(file.Z1_TEST_IMAGE + i + 1)
						ys_test_image = file.arr_image_z1(file.Z1_TEST_IMAGE + i + 1)
						accuracy_score = sess.run(accuracy, feed_dict={x: xs_test_label, y_: ys_test_image})

					print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
				else:
					print("No checkpoint file found")
					return
			time.sleep(TEST_INTERVAL_SECS)


def main():
	test()


if __name__ == '__main__':
	main()

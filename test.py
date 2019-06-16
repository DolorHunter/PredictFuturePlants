#coding:utf-8

import time
import tensorflow as tf
import forward
import backward
TEST_INTERVAL_SECS = 5


# 用于实时监测准确度的测试程序
def test():
	with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
		y_= tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
		y = forward.forward(x, None)

		ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
					print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
				else:
					print("No checkpoint file found")
					return
			time.sleep(TEST_INTERVAL_SECS)


def main():
	test()


if __name__ == '__main__':
	main()

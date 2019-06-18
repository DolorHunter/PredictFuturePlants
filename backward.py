# coding=utf-8

import tensorflow as tf
import file
import forward
import os

BATCH_SIZE = 1                # 定义每轮喂入神经网络多少张图片
LEARNING_RATE_BASE = 0.1      # 学习率
LEARNING_RATE_DECAY = 0.99    # 衰减率
REGULARIZER = 0.001           # 正则化系数
STEPS = 30000                 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99   # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/"  # 模型保存路径
MODEL_NAME = "model"          # 模型se保存文件名


# backward函数中读入图片数据
def backward():
    x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])  # 用placeholder给x占位
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])  # 用placeholder给y_占位
    y = forward.forward(x, REGULARIZER)  # 用前向传播函数计算输出y
    global_step = tf.Variable(0, trainable=False)  # 给轮数计数器global_step赋初值，定义为不可训练

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))  # 调用包含正则化的损失函数loss

    learning_rate = tf.train.exponential_decay(  # 定义指数衰减学习率
        LEARNING_RATE_BASE,
        global_step,
        file.Z1_IMAGE / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()  # 实例化Saver

    with tf.Session() as sess:  # 在with结构中初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):  # 用for循环迭代steps轮
            # 每次读入图片和序号，喂入神经网络进行训练
            xs = file.label_image_z1((STEPS % file.Z1_IMAGE) + 1)
            ys = file.arr_image_z1((STEPS % file.Z1_IMAGE) + 1)
            # 训练，得到损失和步骤，输入为xs, ys
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})

            if i % 100 == 0:
                print("After %d training steps, loss on training batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    return loss_value


def main():
    backward()


if __name__ == '__main__':
    main()

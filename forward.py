# coding=utf-8

import tensorflow as tf

# 定义神经网络的相关参数
INPUT_NODE = 1440000  # 每张图1200*1200个像素点
OUTPUT_NODE = 1440000  # 输出相同像素点1200*1200个
LAYER_NODE = 100  # 隐藏层节点个数1,000,000个
# 1000会超过最大资源


def get_weight(shape, regularizer):
    # 更透过tf.Variable，在训练神经网络时，随机生成参数w
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != 0:  # 如果使用正则化，则将每个变量的损失加入到总损失losses
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    # 搭建神经网络，描述从输入到输出的数据流
    w1 = get_weight([INPUT_NODE, LAYER_NODE], regularizer)
    b1 = get_bias([LAYER_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y   # 因为要对输出使用softmax函数，使它符合概率分布，所以输出y不过value函数

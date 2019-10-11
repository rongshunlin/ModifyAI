#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/10 上午9:14
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  SelfLoss
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf
from numpy.random import RandomState

# define structure
x = tf.placeholder(tf.float32, shape = (None, 2), name = "input_x")
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = "input_y")

w = tf.Variable(tf.random_normal([2,1], stddev = 1, seed = 1), name = "w")
y = tf.matmul(x, w)

# define loss
loss_more = 1
loss_less = 10
loss = tf.reduce_mean(tf.where(tf.greater(y, y_),
                      (y - y_) * loss_more, (y_ - y) * loss_less))
train_op = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

#generate data
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.5] for (x1, x2) in X]

steps = 1000
batch_size = 8
#反复迭代，更新参数
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(steps):
        start = ( i * batch_size) % data_size
        end = min(start + batch_size, data_size)
        sess.run(train_op, feed_dict={x:X[start:end], y_:Y[start:end]})
    print (sess.run(w))


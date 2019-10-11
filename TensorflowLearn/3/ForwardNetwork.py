#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/9 下午9:19
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  ForwardNetwork
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
## define input data
x = tf.placeholder(tf.float32, shape=(None, 2), name="input_x")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="input_y")

## define train weight
w1 = tf.Variable(tf.random_normal(shape=[2,3], stddev = 1, seed = 1), name = "w1")
w2 = tf.Variable(tf.random_normal(shape=[3,1], stddev = 1, seed = 1), name = "w2")

## define network
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

## define loss funcs
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
)
train_steps = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

## 模拟数据集合
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

## 创建Session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    steps = 1000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        ## 更新网络参数
        sess.run(train_steps,
                 feed_dict={x:X[start:end], y_:Y[start:end]})

        if i % 50 == 0:
            ## 隔一段时间，计算所有数据上的交叉上墒
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x:X, y_:Y}
            )
            print ("After %d trainSteps, cross entropy on all data is %g" %(i, total_cross_entropy))
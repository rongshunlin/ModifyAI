#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/9 下午9:06
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  PlaceHolder
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3], stddev=2, seed = 1), name = "w1")
w2 = tf.Variable(tf.random_normal([3,1], stddev=2, seed = 1), name = "w2")

x = tf.placeholder(tf.float32, shape=(None,2), name = "x")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print (sess.run(y, feed_dict={x: [[0.9, 0.7],[0.1, 0.4], [0.5, 0.8]]}))

#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2020/3/9 上午8:13
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  reduce_sum
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf
import numpy as np

indices = [[0, 1, 2], [1,0, 2]]
# indices_sum = tf.reduce_sum(indices)
indice_reshape = tf.reshape(indices, [-1])

flat_positions = tf.range(0,12, dtype=tf.int32) * 12

output = tf.ones([3,1,3])
output_remove = tf.squeeze(output)


with tf.Session() as sess:
    print (sess.run(indice_reshape))
    # print (sess.run(flat_positions))
    # # print (sess.run(output))
    # # print (sess.run(tf.shape(output)))
    # # print (sess.run(output_remove))
    # # print (sess.run(tf.shape(output_remove)))
    # # print (sess.run(output))

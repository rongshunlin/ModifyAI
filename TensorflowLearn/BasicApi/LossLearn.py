#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/19 下午9:46
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  LossLearn
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf

indices = [[0, 1, 2], [1,0, 2]]
one_hot_labels = tf.one_hot(indices=indices, depth=3, dtype=tf.float32)

predict_indecies = [[1, 0, 2], [1,0,2]]
predict_log_labels = tf.one_hot(indices=predict_indecies, depth=3, dtype=tf.float32)
s1 = tf.reshape(tf.range(0, 10, dtype=tf.int32) * 10, [-1, 1]) + tf.reshape(tf.range(0, 10, dtype=tf.int32) * 10, [-1, 1])
with tf.Session() as sess:
    print ("one_hot_labels")
    print (sess.run(one_hot_labels))

    print ("dot values")
    print (sess.run(predict_log_labels * one_hot_labels))


    print ("cross entropy")
    print (sess.run(-tf.reduce_sum(predict_log_labels * one_hot_labels, -1)))

    print ("exp loss")
    # print (sess.run(tf.exp(predict_log_labels)))
    #print (sess.run(tf.reduce_sum(tf.exp(predict_log_labels) * one_hot_labels, -1)))

    print ("test")
    print (sess.run(tf.reshape(s1)))



#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/13 下午1:19
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  VariableScope
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    global_steps = tf.Variable(0, trainable=False)


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(1000):
        print (sess.run(global_steps))
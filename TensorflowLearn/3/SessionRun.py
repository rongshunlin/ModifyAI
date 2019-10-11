#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/8 下午11:05
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  SessionRun
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
resutlt = a + b

#默认都是default_graph
print(a.graph is tf.get_default_graph())

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print(sess.run(resutlt))
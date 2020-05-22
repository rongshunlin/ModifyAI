#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/12/14 上午11:44
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  expand_dim
# @description  ：  仅供学习, 请勿用于商业用途

import numpy as np
import tensorflow as tf

input_ids = [[1,2,3],[2,4,5]]
# input_ids = tf.expand_dims(input_ids, axis=[-1])
new_input_ids = tf.reshape(input_ids,[-1])

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print (sess.run(input_ids))
    print(input_ids.shape)
    print (sess.run(new_input_ids))

#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/25 上午9:07
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  init_rnn
# @description  ：  仅供学习, 请勿用于商业用途

import numpy as np
import tensorflow as tf
X = [1, 2]
state = [0.0, 0.0]
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_rnn = [0.1, -0.1]
b_output = [0.1]
w_outout = [1.0, 2.0]
input = [0.0,0.0]
print ("hellp")

for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_rnn
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_outout) + b_output
    print ("beforre activate:{}, state:{},output:{}".format(before_activation, state, final_output))

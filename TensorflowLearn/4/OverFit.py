#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/10 下午10:48
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  OverFit
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf
from numpy.random import RandomState

# define vars and add vars to collections
def get_weight(shape, lambdaInfo):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lambdaInfo)(var)
    )
    return var

# define network
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8

layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    w = get_weight([in_dimension, out_dimension], 0.0001)
    bias = tf.Variable(tf.constant(0.1, shape = [out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, w) + bias)
    in_dimension = layer_dimension[i]


# define loss
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

# genera data
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.5] for(x1, x2) in X]

steps = 1000
batch_size = 8

# iterate data
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(steps):
        start = (i * batch_size) % data_size
        end = min(start + batch_size, data_size)
        sess.run(train_op, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 50 == 0:
            print ("After %d iterations, total loss is %g"%(i, sess.run(loss, feed_dict={x:X, y_:Y})))
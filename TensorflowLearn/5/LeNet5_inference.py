#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/16 上午7:16
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  LeNet5_inference
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf

REGULARIZATION_RATE = 0.0001
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABLES = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

def get_weight_variable(name, shape, regularizer):
    weights = tf.get_variable(name, shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights

def inference(input_tensor, train, regularizer):
    # define layer1
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = get_weight_variable("conv1_weight", [5, 5, 1, 6], None)
        cov1_biases = get_weight_variable("cov1_biases", [6], None)
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1, 1, 1, 1], padding = "SAME")
        layer1_bias = tf.nn.bias_add(conv1, cov1_biases)
        relu1 = tf.nn.relu(layer1_bias)
        # 32 * 32 * 1 => 28 * 28 * 6
        tf.compat.v1.logging.info("Shape of relu1:{}".format(str(relu1.shape)))

    # define layer2
    # 28 * 28 * 6 => 14 * 14 * 6
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        tf.compat.v1.logging.info("Shape of layer2_pool:{}".format(str(pool1.shape)))

    # define layer3
    # 14 * 14 * 6 => 10 * 10 * 16
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = get_weight_variable("conv2_weight", [5, 5, 6, 16], None)
        conv2_biases = get_weight_variable("conv2_biases", [16], None)
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding = "VALID")
        layer3_bias = tf.nn.bias_add(conv2, conv2_biases)
        layer3_actived_conv = tf.nn.relu(layer3_bias)
        tf.compat.v1.logging.info("Shape of layer3_actived_conv:{}".format(str(layer3_actived_conv.shape)))

    # define layer4
    # 10 * 10 * 16 => 5 * 5 * 16
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(layer3_actived_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        tf.compat.v1.logging.info("Shape of pool2:{}".format(str(pool2.shape)))

    # define layer5
    # 5 * 5 * 16 => 1 * 1 * 120
    # pool_shape =>[100，5,，5， 16]
    pool_shape = pool2.get_shape().as_list()
    print ("pool_shape :" + str(pool_shape))
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    pool2_reshape = tf.reshape(pool2, [-1, nodes])
    tf.compat.v1.logging.info("pool2_reshape:{}".format(str(pool2_reshape.shape)))

    with tf.variable_scope('layer5-fc1'):
        fc1_weight = get_weight_variable("fc1_weight", [nodes, 120], regularizer)
        fc1_biases = get_weight_variable("fc1_biases", [120], None)
        fc1 = tf.nn.relu(tf.matmul(pool2_reshape, fc1_weight) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
        tf.compat.v1.logging.info("Shape of fc1:{}".format(str(fc1.shape)))

    with tf.variable_scope('layer6-fc2'):
        fc2_weight = get_weight_variable("fc2_weight", [120, 84], regularizer)
        fc2_biases = get_weight_variable("fc2_biases", [84], None)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weight) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)
        tf.compat.v1.logging.info("Shape of fc2:{}".format(str(fc2.shape)))

    with tf.variable_scope('layer6-fc3'):
        fc3_weight = get_weight_variable("fc3_weight", [84, 10], regularizer)
        fc3_biases = get_weight_variable("fc3_biases", [10], None)
        fc3 = tf.matmul(fc2, fc3_weight) + fc3_biases
        if train:
            fc3 = tf.nn.dropout(fc3, 0.5)
        tf.compat.v1.logging.info("Shape of fc3:{}".format(str(fc3.shape)))
    return fc3
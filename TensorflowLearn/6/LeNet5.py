#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/14 下午10:47
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  LeNet5
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf

REGULARIZATION_RATE = 0.0001

def get_weight_variable(name, shape, regularizer):
    weights = tf.get_variable(name, shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights

def inference():
    x = tf.placeholder(tf.float32, [None, 32, 32], name = "input_x")
    y_ = tf.placeholder(tf.float32, [None, 10], name = "output_y")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # define layer1
    layer1_filter_weight = get_weight_variable("layer1_filter_weight", [5, 5, 1, 6], regularizer)
    layer1_biases = get_weight_variable("layer1_biases", [6], regularizer)
    conv1 = tf.nn.conv2d(x, layer1_filter_weight, strides=[1, 1, 1, 1], padding = "valid")
    layer1_bias = tf.nn.bias_add(conv1, layer1_biases)
    layer1_actived_conv = tf.nn.relu(layer1_bias)
    # 32 * 32 * 1 => 28 * 28 * 6
    tf.compat.v1.logging.info("Shape of layer1_actived_conv:{}".format(str(layer1_actived_conv.shape)))

    # define layer2
    # 28 * 28 * 6 => 14 * 14 * 6
    layer2_pool = tf.nn.max_pool(layer1_actived_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    tf.compat.v1.logging.info("Shape of layer2_pool:{}".format(str(layer2_pool.shape)))

    # define layer3
    # 14 * 14 * 6 => 10 * 10 * 16
    layer3_filter_weight = get_weight_variable("layer3_filter_weight", [5, 5, 6, 16], regularizer)
    layer3_biases = get_weight_variable("layer3_biases", [16], regularizer)
    conv3 = tf.nn.conv2d(layer2_pool, layer3_filter_weight, strides=[1, 1, 1, 1], padding = "valid")
    layer3_bias = tf.nn.bias_add(conv3, layer3_biases)
    layer3_actived_conv = tf.nn.relu(layer3_bias)
    tf.compat.v1.logging.info("Shape of layer3_actived_conv:{}".format(str(layer3_actived_conv.shape)))

    # define layer4
    # 10 * 10 * 16 => 5 * 5 * 16
    layer4_pool = tf.nn.max_pool(layer3_actived_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    tf.compat.v1.logging.info("Shape of layer4_pool:{}".format(str(layer4_pool.shape)))

    # define layer5
    # 5 * 5 * 16 => 1 * 1 * 120
    layer5_filter_weight = get_weight_variable("layer5_filter_weight", [5, 5, 16, 120], regularizer)
    layer5_biases = get_weight_variable("layer5_biases", [120], regularizer)
    conv5 = tf.nn.conv2d(layer4_pool, layer5_filter_weight, strides=[1, 1, 1, 1], padding = "valid")
    layer5_bias = tf.nn.bias_add(conv5, layer5_biases)
    layer5_actived_conv = tf.nn.relu(layer5_bias)
    tf.compat.v1.logging.info("Shape of layer3_actived_conv:{}".format(str(layer5_actived_conv.shape)))









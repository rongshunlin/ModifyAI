#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/14 上午8:37
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  mnist_eval
# @description  ：  仅供学习, 请勿用于商业用途

import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name = "y-output")
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}

        y = mnist_inference.inference(x, None)
        correnct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correnct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print ("After %s training steps, validation acc = %g"%(global_step, accuracy_score))
                else:
                    print ("no checkpoint ")
                    return

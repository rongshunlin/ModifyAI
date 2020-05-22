#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/13 下午8:11
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  mnist_train
# @description  ：  仅供学习, 请勿用于商业用途

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import mnist_inference
import LeNet5_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

# model's path and file name
MODEL_SAVE_PATH = "/Users/rongshunlin/script/models"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [None, 784], name = "input_x")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name = "output_y")
    tf.compat.v1.logging.info("Shape of y_:{}".format(str(y_.shape)))

    reshape_x = tf.reshape(x, [BATCH_SIZE, 28, 28, 1])
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # y = mnist_inference.inference(x, regularizer)
    y = LeNet5_inference.inference(reshape_x, True, regularizer)
    tf.compat.v1.logging.info("Shape of y:{}".format(str(y.shape)))

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_averages.apply(tf.trainable_variables())
    tf.compat.v1.logging.info("Before cross entropy")
    tf.compat.v1.logging.info("Shape of y:{}, Shape of y_:{}, shape of argmax:{}".format(str(y.shape), str(y_.shape), str(tf.argmax(y_, 1).shape)))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_: ys})
            if i % 50 == 0:
                print ("After %d train steps, loss on training batch is %g."%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)


def main(argv = None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()
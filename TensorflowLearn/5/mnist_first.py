#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/12 下午10:23
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  mnist_first
# @description  ：  仅供学习, 请勿用于商业用途
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# define parameters
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weight1, bias1,
              weight2, bias2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + bias1)
        return tf.matmul(layer1, weight2) + bias2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(bias1))
        return tf.matmul(layer1, avg_class.average(weight2) + avg_class.average(bias2))

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = "input_x")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = "output_y")

    w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))

    w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev= 0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
    y = inference(x, None, w1, bias1, w2, bias2)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, w1, bias1, w2, bias2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(w1) + regularizer(w2)
    loss = cross_entropy_mean + regularization
    learn_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step = global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 100 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d train steps, valid acc is %g" %(i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs, y_:ys})
    test_acc, steps = sess.run([accuracy, global_step], feed_dict=test_feed)
    print("After %d train steps, test acc is %g" % (steps, test_acc))


def main(argv = None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()




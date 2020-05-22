#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/17 下午10:52
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  read_tf
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf

filename = "/Users/rongshunlin/xiaomiData/tensorflow_learn/output.tfrecords"
reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer([filename])
_, seralized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    seralized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'labels': tf.FixedLenFeature([], tf.int64)
    })

images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['labels'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
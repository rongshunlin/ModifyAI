#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/10/17 下午10:27
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  create_tf
# @description  ：  仅供学习, 请勿用于商业用途

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets("/Users/rongshunlin/xiaomiData/tensorflow_learn", dtype=tf.uint8, one_hot=True)

images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = "/Users/rongshunlin/xiaomiData/tensorflow_learn/output.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'labels': _int64_feature(np.argmax(labels[index])),
        'image_raw':_bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()

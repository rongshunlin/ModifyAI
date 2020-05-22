#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/12/14 下午12:00
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  BertModelTest
# @description  ：  仅供学习, 请勿用于商业用途
from BertModel import embedding_lookup
from BertModel import embedding_lookup_with_mask
from BertModel import  create_attention_mask_from_input_mask
import  tensorflow as tf

def test_embedding_lookup():
    input_ids = tf.Variable([[1, 2, 3], [4, 5,1]], dtype=tf.int32)
    print (input_ids.shape)
    (input_ids, embedding_table) = embedding_lookup(input_ids, vocab_size=10, embedding_size=10)
    print ("input_ids size")
    print (input_ids.shape)
    return input_ids

def test_embedding_lookup_mask():
    input_ids = tf.Variable([[1, 2, 3], [4, 5, 1]], dtype=tf.int32)
    input_ids_mask = tf.Variable([[1, 2, 3], [0, 0,0]], dtype=tf.float32)

    (embedding, embeding_table) = embedding_lookup_with_mask(input_ids, input_mask = input_ids_mask,vocab_size=10)
    return input_ids

def test_create_attention_mask_from_input_mask():
    input_ids = tf.Variable([[1, 2, 3], [4, 5, 1],[1,1,1]], dtype=tf.int32)
    print ("input_ids dim :{}".format(input_ids.shape.ndims))
    input_ids_mask = tf.Variable([[1, 2, 3], [0, 0,0],[0,0,0]], dtype=tf.float32)
    to_mask, mask = create_attention_mask_from_input_mask(input_ids, input_ids_mask)
    return mask, to_mask

def test_tf_concat():
    input_ids = tf.Variable([[1, 2, 3], [4, 5, 1],[1,1,1]], dtype=tf.int32)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # input_ids = test_embedding_lookup()
    mask, to_mask = test_create_attention_mask_from_input_mask()
    # input_ids = test_embedding_lookup_mask()
    ## 创建Session
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # print (sess.run(input_ids))
        print (sess.run(mask))
        print (mask.shape)
        print ("------------")
        print(sess.run(to_mask))
        print(to_mask.shape)
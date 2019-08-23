#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/8/10 下午10:20
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  model.py
# @description  ：  仅供学习, 请勿用于商业用途

import os
import tensorflow as tf
import numpy as np
import data
import datetime
from text_cnn import ModelConfig, TextCNNModel
from tensorflow.contrib import learn

flags = tf.flags
FLAGS = flags.FLAGS

# 数据路径
flags.DEFINE_string("positive_data_file", "./data_set/polarity.pos", "splited by ,")
flags.DEFINE_string("negative_data_file", "./data_set/polarity.neg", "splited by ,")
flags.DEFINE_string("pred_data", "None", "splited by ,")
flags.DEFINE_string("model_dir", "./data/model/", "output model dir")
flags.DEFINE_string("output_dir", "./data/model/", "evaluate output dir")
flags.DEFINE_string("vocab", None, "vocab file")

# cnn 参数
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("drop_rate", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_integer("max_seq_length", 64, "Maximum sequence length")
flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0")

# 训练参数
flags.DEFINE_bool("is_train", True, "Whether to run training.")
flags.DEFINE_bool("is_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("is_predict", False, "Whether to run prediction.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
flags.DEFINE_integer("num_train_steps", 100000, "Train steps")
flags.DEFINE_integer("keep_checkpoint_max", 20, "Max keep checkpoints")
flags.DEFINE_integer("save_summary_steps", 1000, "Step intervals to save summary")
flags.DEFINE_integer("log_step_count_steps", 1000, "Step intervals to log step info")
flags.DEFINE_integer("save_checkpoints_steps", 500, "Step intervals to save checkpoints")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")


def preprocess():
    data_info = data.DataSet(FLAGS.positive_data_file, FLAGS.negative_data_file)
    x_text, y = data_info.x_text, data_info.y

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    tf.logging.info("Shape of X :{}".format(str(x.shape)))

    # Random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    # Init model config
    model_config = ModelConfig(
        embedding_dim=FLAGS.embedding_dim,
        filter_sizes=FLAGS.filter_sizes,
        num_filters=FLAGS.num_filters,
        dropout_rate=FLAGS.drop_rate,
        l2_reg_lambda=FLAGS.l2_reg_lambda,
        max_seq_length=max_document_length,
        vocab_size=len(vocab_processor.vocabulary_),
        label_size=2
    )
    tf.logging.info("Vocabulary size: {:d}".format(len(vocab_processor.vocabulary_)))
    tf.logging.info("Train/dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    tf.logging.info("*******Init Model CONFIG*************")
    tf.logging.info(model_config.to_string())
    return x_train, y_train, vocab_processor, x_dev, y_dev, model_config


def train(x_train, y_train, vocab_processor, x_dev, y_dev, model_config):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextCNNModel(
                config=model_config,
                is_training=FLAGS.is_train
            )
            # Define Training proceduce
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Checkpoint directory, Tensorflow assumes this directioon already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(FLAGS.output_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep_checkpoint_max)

            # Write vocabulary
            vocab_processor.save(os.path.join(FLAGS.output_dir, "vocab"))

            # Initialize all variables
            summary_writer = tf.summary.FileWriter('./log/', sess.graph)
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A singel training step
                :param x_batch:
                :param y_batch:
                :return:
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                tf.logging.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                tf.logging.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # Generate batches
            batches = data.DataSet.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop, For each batch ..
            for batch in batches:
                x_batch, y_batch = zip(*batch)
            #     train_step(x_batch, y_batch)
            #     current_step = tf.train.global_step(sess, global_step)
            #
            #     if current_step % FLAGS.save_checkpoints_steps == 0:
            #         tf.logging.info("\nEvaluation:")
            #         dev_step(x_dev, y_dev)
            #     if current_step % FLAGS.save_checkpoints_steps == 0:
            #         path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #         tf.logging.info("Saved model checkpoint to {}\n".format(path))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    x_train, y_train, vocab_processor, x_dev, y_dev, config = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, config)


if __name__ == "__main__":
    tf.app.run()

#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/12/14 下午12:00
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  BertModel
# @description  ：  仅供学习, 请勿用于商业用途
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 var_prefix="",
                 si_vocab_size=1024,
                 si_hidden_size=256,
                 intent_vocab_size=512,
                 intent_hidden_size=128
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.var_prefix = var_prefix
        self.si_vocab_size = si_vocab_size
        self.si_hidden_size = si_hidden_size
        self.intent_vocab_size = intent_vocab_size
        self.intent_hidden_size = intent_hidden_size

    @classmethod
    def from_dict(cls,  json_object):
        config = BertConfig(vocab_size=None)
        for(key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

    def to_josn_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertModel(object):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 si_seq_ids=None,
                 si_seq_mask=None,
                 si_vocab_size=None,
                 si_hidden_size=None,
                 intent_ids=None,
                 intent_mask=None,
                 intent_vocab_size=None,
                 intent_hidden_size=None,
                 prejudge_feats=None,
                 use_one_hot_embeddings=False,
                 scope=None,
                 is_only_char_feat=True):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        # input_shape's rank should be 2
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        # input ids => [batch_szie, 32]
        # input_mask => [batch_szie, 32]
        # token_type_ids => [batch_szie, 32]
        # si_seq_ids => [batch_szie, 32 * 5]
        # si_seq_mask => [batch_szie, 32 * 5]
        # intent_ids => [batch_size, 128]
        # intent_mask => [batch_size, 128]

        with tf.variable_scope(scope, default_name=config.var_prefix + "bert"):
            # hidden_size: Size of the encoder layers and the pooler layer.
            with tf.variable_scope("embeddings"):
                # input_ids=>[batch_size, seq_length]
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    use_one_hot_embedding=use_one_hot_embeddings
                )
                # embedding_out=>[batch_size, seq_length, embedding_size]
                if not is_only_char_feat:
                    # si_seq_ids => [batch_size, seq_length, max_slot_length]
                    si_seq_ids = tf.reshape(si_seq_ids, [batch_size, seq_length, -1])
                    si_seq_mask = tf.reshape(si_seq_ids, [batch_size, seq_length, -1])
                    (self.embedding_si_output, self.embedding_si_table) = embedding_lookup_with_mask(
                        input_ids=si_seq_ids,
                        input_mask=si_seq_mask,
                        vocab_size=si_vocab_size,
                        embedding_size=si_hidden_size,
                        word_embedding_name="si_embeddings"
                    )
                    # embedding_si_output => [batch_size, seq_length, embedding_size]

                    #embedding_input_output => [batch_size, 1, embedding_size]
                    (self.embedding_input_output, self.embedding_intent_table) = embedding_lookup_with_mask(
                        input_ids=intent_ids,
                        input_mask=intent_mask,
                        vocab_size=intent_vocab_size,
                        embedding_size=intent_hidden_size,
                        word_embedding_name="si_embeddings"
                    )
                    self.embedding_intent_output = tf.reduce_sum(self.embedding_input_output, axis=-2)
                    # embedding_intent_output => [batch_size, embedding_size]

                    self.embedding_output = tf.add(self.embedding_output, self.embedding_si_output)

                # add positional embedding and token types embeddings, then layer normalize and perform drouout
                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                # embedding_out=>[batch_size, seq_length, embedding_size]
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                # attention mask => [batch_size, from_seq_length, to_seq_length]
                # attention mask => [batch_size, seq_length, seq_length]

                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # embedding_out=>[batch_size, seq_length, embedding_size]
                # embedding_size == hidden_size
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

                self.sequence_output = self.all_encoder_layersp[-1]

                with tf.variable_scope("pooler"):
                    first_token_tensor = tf.squeeze(self.sequence_output[:,0:1,:], axis=1)
                    # embedding_intent_outpout=>[batch_size,embedding_size]
                    # first_token_tensor =>[batch_size,embedding_size] ?
                    if not is_only_char_feat:
                        first_token_tensor = tf.concat(
                            [first_token_tensor, self.embedding_intent_output], axis = -1)
                    # first_token_tensor =>[batch_size, 2 * embedding_size]
                    self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        config.hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=create_initializer(config.initializer_range))

def create_initializer(initializer_range=0.02):
    """Creates a `truncated normal initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known activation.
    """

    # We assume that anything that's not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)

def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.

    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def transformer_model(
        input_tensor,
        attention_mask=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn=gelu,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        do_return_all_layers=False):

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    # attention_head_size = 1568/12 = 128
    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=[3])
    batch_size = input_shape[0]
    # seq_length = 32
    seq_length = input_shape[1]
    # inpuit_width = hidden_sisze = 1568
    input_width = input_shape[2]

    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    prev_output = reshape_to_matrix(input_tensor)
    #prev_output => [batch_size*seq_length, hidden_size]

    all_lay_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_hidden_layers,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length
                    )
                    attention_heads.append(attention_head)
                    #attention_heads => [batch_size*seq_length, hidden_size]

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    attention_output = tf.concat(attention_heads, axis = -1)

                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=creat_initializer(initializer_range)
                    )
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    # residual connect
                    attention_output = layer_norm(attention_output + layer_input)

            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=creat_initializer(initializer_range)
                )

            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=creat_initializer(initializer_range))
                layer_output = dropout(layer_output)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_lay_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_lay_outputs:
            # layer_output [batch_size*seq_length, hidden_size] = > [batch_size, seq_length, hidden_size]
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output

def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor,
                                   [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2,3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2,3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = from_shape[2]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)
    # from_tensor_2d => [batch_size * seq_length, hidden_size]

    # Scalar dimensions referenced here:
    #   B = batch_size (number of sequences)
    #   F = `from_tensor` sequence_length
    #   T = `to_tensor` sequence_length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # `query_layer` = [B*F, N*H]
    # N*H => hidden size
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=creat_initializer(initializer_range)
    )

    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=creat_initializer(initializer_range)
    )

    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=creat_initializer(initializer_range)
    )

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, from_seq_length, size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head)

    # calculate attention scores
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_sores = tf.multiply(attention_scores, 1.0/math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # attention_masks => [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        addr = (1.0 - tf.cast(attention_mask, tf.float32)) * -100000.0
        attention_scores += addr

    attention_probs = tf.nn.softmax(attention_scores)
    # attention_probs = [B, N, F, T]
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # value_layer = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # value_layer = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # context_layer = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # context_layer = [B, F, N, H]
    context_layer = transpose_for_scores(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(context_layer, batch_size, from_seq_length, num_attention_heads * size_per_head)

    return context_layer

def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))

    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

def create_attention_mask_from_input_mask(from_tensor, to_mask):
    # from tensor's rank should be 2 or 3
    from_shape = get_shape_list(from_tensor, expected_rank=[2,3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    tf.logging.info("from shape:{}".format(from_shape))
    to_shape = get_shape_list(to_mask, expected_rank=[2])
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    tf.logging.info("to mask:{}".format(to_mask))
    broad_cast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    mask = broad_cast_ones * to_mask
    return mask

def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embedding",
                            use_position_embedding=True,
                            position_embedding_name="position_embedding",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1
                            ):
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor
    #output=>[batch_size, seq_length, width]

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if `use_token_type` is True")

        token_type_table = tf.get_variable(
             name=token_type_embedding_name,
             shape=[token_type_vocab_size, width],
             initializer_range=creat_initializer(initializer_range)
        )
        # token_type_ids => [batch_size, seq_length]
        flat_token_type_ids = tf.reshape(token_type_ids,[-1])
        # flat_token_types_ids => [batch_size * seq_length, 1]
        one_hot_ids = tf.one_hot(flat_token_type_ids,depth=token_type_vocab_size)
        # one_hot_ids =>[batch_size * seq_length, token_type_vocab_size]
        # token_type_embedding => [batch_size * seq_length, witdth]
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])

        output += token_type_embeddings

    if use_position_embedding:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=creat_initializer(initializer_range)
            )
            #https://www.jianshu.com/p/71e6ef6c121b 参考链接
            position_embeddings = tf.slice(full_position_embeddings, [0,0],
                                           [seq_length, -1])
            # position_embeddings => [seq_length, witdth]
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            # position_embeddings => [1, seq_length, width]
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            # boardcast among the first dimentsions
            output += position_embeddings

    output =layer_norm_and_dropout(output, dropout_prob)
    return output

def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
        input_tensor: float Tensor.
        dropout_prob: Python float. The probability of dropping out a value (NOT of
            *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output

def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def layer_norm_and_dropout(input_tensor, droput_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, droput_prob)
    return output_tensor

# 把word_slot_length合并了
def embedding_lookup_with_mask(input_ids,
                               input_mask,
                               vocab_size,
                               embedding_size=128,
                               word_embedding_name="word_embeddings"):
    tf.logging.info("input_ids shape :{}".format(input_ids.shape))
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-2])
        input_mask = tf.expand_dims(input_mask, axis=[-2])
    #intent_ids =>[batch_size, 1, 128]
    embedding_table = tf.get_variable(name=word_embedding_name, shape=[vocab_size, embedding_size],
                                      initializer=tf.random_normal_initializer(0., embedding_size ** -0.5))
    #embedding => [batch_size, seq_length,  max_slot_length, embedding_size]

    tf.logging.info("input_ids shape :{}".format(input_ids.shape))
    embedding = tf.gather(embedding_table, input_ids)
    #intent_ids_embeddings =>[batch_size, 1, 128, embedding_dim]
    tf.logging.info("step 1 embeddng shape.{}".format(embedding.shape))
    #input_mask=> [batch_size, seq_length,  max_slot_length, 1]
    embedding *= tf.expand_dims(input_mask, -1)
    #embedding =>[batch_size, seq_length,  max_slot_length, embedding_size]
    embedding = tf.reduce_sum(embedding, axis=[-2])
    tf.logging.info("step 2 embeddng shape.{}".format(embedding.shape))
    #embedding => [batch_size, seq_length, embedding_size]
    #intent_ids =>[batch_size,1, embedding_dims]
    return embedding, embedding_table


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embedding=False):
    #input_ids =>[batch_size, seq_length]
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])
    #input_ids =>[batch_size, seq_length, 1]
    print ("first input" + str(input_ids.shape))
    embedding_table = tf.get_variable(name=word_embedding_name,
                                      shape=[vocab_size, embedding_size],
                                      initializer=creat_initializer(initializer_range))
    #flat_input_ids =>[batch_size * seq_length, 1]
    flat_input_ids = tf.reshape(input_ids, [-1])
    print ("flat_input_ids" + str(flat_input_ids.shape))

    if use_one_hot_embedding:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        #one_hot_input_ids =>[batch_size * seq_length, vocab_size]
        output = tf.matmul(one_hot_input_ids, embedding_table)
        #output =>[batch_size * seq_length, embedding_size]
    else:
        output = tf.gather(embedding_table, flat_input_ids)
        #output =>[batch_size * seq_length, embedding_size]

    input_shape = get_shape_list(input_ids)
    #input_shape =>[batch_size, seq_length, 1]

    print("base size:" + str(input_shape))
    #output will be reshape as [batch_size, seq_length, embedding_size]
    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def creat_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(initializer_range)

def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()
    print ("shape is")
    print (shape)
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib import layers

from model.utils import utils


def vgg(inputs, num_output, is_train = True, scope = "vgg", weight_decay = 0.0005):
    assert num_output % 8 == 0
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
        net = layers.repeat(inputs, 2, layers.conv2d, num_output//8, [3, 3],
                            activation_fn=tf.nn.relu, weights_regularizer = layers.l2_regularizer(weight_decay),
                            biases_initializer = tf.zeros_initializer(), scope="conv1")
        net = layers.max_pool2d(net, [2, 2], scope="pool1")
        net = layers.repeat(net, 2, layers.conv2d, num_output//4, [3, 3],
                            activation_fn=tf.nn.relu, weights_regularizer = layers.l2_regularizer(weight_decay),
                            biases_initializer = tf.zeros_initializer(), scope="conv2")
        net = layers.max_pool2d(net, [2, 2], scope="pool2")
        net = layers.repeat(net, 2, layers.conv2d, num_output//2, [3, 3],
                            activation_fn=tf.nn.relu, weights_regularizer = layers.l2_regularizer(weight_decay),
                            biases_initializer = tf.zeros_initializer(), scope="conv3")
        net = layers.max_pool2d(net, [2, 2], scope="pool3")
        net = layers.repeat(net, 2, layers.conv2d, num_output, [3, 3], activation_fn=tf.nn.relu,
                            weights_regularizer = layers.l2_regularizer(weight_decay),
                            biases_initializer = tf.zeros_initializer(), scope="conv4")
        net = layers.max_pool2d(net, [2, 2], scope="pool4")
        net = layers.flatten(net, scope="fc5")
        return net

def mobile_net(inputs, num_output, width_multiplier = 1, is_train = True, scope = "mobile_net", weight_decay = 0.0005):
    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  width_multiplier,
                                  scope,
                                  is_train,
                                  downsample = False):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1
        depthwith_conv = layers.separable_conv2d(inputs,
                                                 num_outputs = None, # If is None, then we skip the pointwise convolution stage.
                                                 stride = _stride,
                                                 depth_multiplier = 1,
                                                 kernel_size = [3, 3],
                                                 scope = scope + "/depthwise_conv")
        depthwith_conv = layers.batch_norm(depthwith_conv, trainable=is_train, scope=scope + "/dw_batch_norm")
        depthwith_conv = tf.nn.relu(depthwith_conv, name=scope + "/dw_relu")
        pointwith_conv = layers.conv2d(depthwith_conv,
                                       num_outputs = num_pwc_filters,
                                       kernel_size=[1, 1],
                                       scope = scope + "/pointwise_conv")
        pointwith_conv = layers.batch_norm(pointwith_conv, trainable=is_train, scope=scope + "/pw_batch_norm")
        pointwith_conv = tf.nn.relu(pointwith_conv, name=scope + "/pw_batch_relu")
        return pointwith_conv

    assert num_output % 8 == 0
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = layers.conv2d(inputs, round(num_output // 8 * width_multiplier), [3, 3], stride=2,
                            weights_regularizer = layers.l2_regularizer(weight_decay),
                            normalizer_fn = layers.batch_norm,
                            activation_fn = tf.nn.relu, padding = "SAME", trainable = is_train, scope = "conv1")
        net = _depthwise_separable_conv(net, num_output // 4, width_multiplier, is_train = is_train, scope = "conv_ds_2")
        net = _depthwise_separable_conv(net, num_output // 2, width_multiplier, is_train = is_train, downsample = True, scope = "conv_ds_3")
        net = _depthwise_separable_conv(net, num_output // 2, width_multiplier, is_train = is_train, scope = "conv_ds_4")
        net = _depthwise_separable_conv(net, num_output, width_multiplier, is_train=is_train, downsample=True, scope="conv_ds_5")
        net = _depthwise_separable_conv(net, num_output, width_multiplier, is_train=is_train, scope="conv_ds_6")
        net = _depthwise_separable_conv(net, num_output, width_multiplier, is_train=is_train, downsample=True, scope="conv_ds_7")

        net = _depthwise_separable_conv(net, num_output, width_multiplier, is_train=is_train, scope="conv_ds_8")
        net = _depthwise_separable_conv(net, num_output, width_multiplier, is_train=is_train, scope="conv_ds_9")
        net = layers.flatten(net, scope="fc10")
        return net


def resnet(inputs, num_output, is_train = True, scope = "resnet", weight_decay = 0.0005):

    def conv2d_same(data,
                    num_outputs,
                    kernel_size,
                    stride,
                    is_train=True,
                    weight_decay=0.0005,
                    activation_fn=tf.nn.relu,
                    normalizer_fc=True):
        if stride == 1:
            data = layers.conv2d(inputs=data,
                                 num_outputs=num_outputs,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 weights_regularizer=layers.l2_regularizer(weight_decay),
                                 activation_fn=None,
                                 padding="SAME")
        else:
            # 默认使用正方形filter
            # feature map size = (input_size + 2 * padding - filter_size) / stride + 1
            # if padding * 2  == filter_size - 1: map_size = (input_size - 1) / stride + 1 = ⌈input_size / stride⌉
            pad_total = kernel_size - 1
            pad_begin = pad_total // 2
            pad_end = pad_total - pad_begin
            data = tf.pad(data, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
            data = layers.conv2d(inputs=data,
                                 num_outputs=num_outputs,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 weights_regularizer=layers.l2_regularizer(weight_decay),
                                 activation_fn=None,
                                 padding='VALID')
        if normalizer_fc:
            data = tf.layers.batch_normalization(data, training=is_train)
        if activation_fn:
            data = activation_fn(data)
        return data

    def bottle_net(data, output_depth, is_train, stride=1, weight_decay=0.0005):
        depth = int(data.shape[-1])

        if depth == output_depth:
            shortcut_tensor = data
        else:
            shortcut_tensor = conv2d_same(data, output_depth, 1, stride=1, is_train=is_train, weight_decay=weight_decay,
                                          activation_fn=None, normalizer_fc=True)

        data = conv2d_same(data, output_depth // 4, 1, 1, weight_decay=weight_decay, is_train=is_train)
        data = conv2d_same(data, output_depth // 2, 3, stride, weight_decay=weight_decay, is_train=is_train)
        data = conv2d_same(data, output_depth, 1, 1, weight_decay=weight_decay, is_train=is_train, activation_fn=None)

        data = data + shortcut_tensor
        data = tf.nn.relu(data)
        return data

    def create_block(data, output_depth, block_nums, init_stride=1, is_train=True, scope="block", weight_decay=0.0005):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            data = bottle_net(data, output_depth, is_train=is_train, weight_decay=weight_decay, stride=init_stride)
            for i in range(1, block_nums):
                data = bottle_net(data, output_depth, is_train=is_train, weight_decay=weight_decay, )
            return data

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        data = conv2d_same(inputs, 64, 3, 1, is_train = is_train, normalizer_fc = True, weight_decay = weight_decay)
        data = layers.max_pool2d(data, 3, 2, padding="SAME", scope = "pool1")
        data = create_block(data, 128, 3, init_stride=1, is_train = is_train, weight_decay = weight_decay, scope = "block1")
        data = create_block(data, 256, 3, init_stride=1, is_train = is_train, weight_decay = weight_decay, scope = "block2")
        data = layers.avg_pool2d(data, 4)
        data = layers.conv2d(data, num_output, 1, activation_fn=None, scope="final_conv")
        data = tf.layers.flatten(data, name="fc")
        return data


def cnn_lstm(inputs,
             filter_sizes, num_filters,
             length, lstm_size, lstm_num_layers, dropout_rate, lstm_dropout_rate,
             scope="cnn_lstm"):
    channels = inputs.shape[1]
    inputs = tf.reshape(inputs, shape=(-1, inputs.shape[2], inputs.shape[3]))
    pooled_output = []
    for filter_size in filter_sizes:
        conv = utils.convolution_1d(inputs, filter_size, num_filters, name_scope=scope + "_conv")
        pooled = utils.max_pool_1d(conv, filter_size, name_scope=scope + "_max_pooling")
        pooled_output.append(pooled)
    # pooled_output = [batch_size * channels, num_filters] * len(num_filters)
    num_filters_total = len(filter_sizes) * num_filters
    pooled_output = tf.reshape(tf.concat(pooled_output, axis=1), shape=(-1, num_filters_total))
    print(pooled_output)
    pooled_output = tf.layers.dropout(pooled_output, rate= 1- dropout_rate)

    pooled_output = tf.reshape(pooled_output, shape=(-1, channels, num_filters_total))
    with tf.name_scope(scope + "_LSTM"):
        fw_cell, bw_cell = [utils.create_lstm_cell(lstm_size, lstm_dropout_rate, lstm_num_layers,
                                                   reuse=tf.AUTO_REUSE) for _ in range(2)]
        lstm_output, lstm_state = utils.dynamic_origin_bilstm_layer(fw_cell, bw_cell, pooled_output, length)
        output_mask = tf.sequence_mask(length, pooled_output.shape[1], dtype=tf.float32)
        output_mask = tf.expand_dims(output_mask, axis=-1)
        lstm_output_mask = lstm_output * output_mask
    return lstm_output_mask, lstm_state


def hierarchical_1d_CNN(inputs, length,
                        filter_sizes, num_filters,
                        filter_sizes_hierarchical, num_filters_hierarchical,
                        dropout_rate, position_embedding_dim = 32, scope = "hierarchical_1dCNN"):
    """
    :param inputs:  [batch_size, channels, length, width]
    :param filter_sizes:
    :param num_filters:
    :param filter_sizes_hierarchical:
    :param num_filters_hierarchical:
    :param dropout_rate:
    :param scope:
    :return:
    """
    channels = inputs.shape[1]
    inputs = tf.reshape(inputs, shape=(-1, inputs.shape[2], inputs.shape[3]))
    pooled_output = []
    for filter_size in filter_sizes:
        conv = utils.convolution_1d(inputs, filter_size, num_filters, name_scope=scope + "_conv_inner")
        pooled = utils.max_pool_1d(conv, filter_size, name_scope=scope + "_max_pooling_inner")
        pooled_output.append(pooled)
    # pooled_output = [batch_size * channels, num_filters] * len(num_filters)
    num_filters_total = len(filter_sizes) * num_filters
    pooled_output = tf.reshape(tf.concat(pooled_output, axis=1), shape=(-1, num_filters_total))
    pooled_output = tf.layers.dropout(pooled_output, rate = 1 - dropout_rate)

    pooled_output = tf.reshape(pooled_output, shape = (-1, channels, num_filters_total))
    position_encoding = utils.position_embeded(channels, num_filters_total, name="position_embedding")
    mask = tf.sequence_mask(length, channels, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    position_encoding *= mask
    pooled_output += position_encoding
    pooled_output = utils.layer_norm(pooled_output, name="layer_normalization")

    hierarchical_pooled_output = []
    for filter_size in filter_sizes_hierarchical:
        conv = utils.convolution_1d(pooled_output, filter_size, num_filters_hierarchical, name_scope=scope + "_conv_outer")
        pooled = utils.max_pool_1d(conv, filter_size, name_scope=scope + "_max_pooling_outer")
        hierarchical_pooled_output.append(pooled)
    hierarchical_num_filters_total = len(filter_sizes_hierarchical) * num_filters_hierarchical
    hierarchical_pooled_output = tf.reshape(tf.concat(hierarchical_pooled_output, axis=1), shape=(-1, hierarchical_num_filters_total))
    return hierarchical_pooled_output


def mlp(data,
        num_layers,
        hidden_size,
        dropout_rate,
        is_train = True,
        weight_decay = 0.0005,
        activation_fn = tf.nn.relu,
        normalizer_fc = True,
        scope = "mlp"):
    for i in range(1, num_layers+1):
        with tf.variable_scope(scope + "_" + str(i), reuse=tf.AUTO_REUSE):
            data = tf.layers.dense(inputs = data,
                                   units = hidden_size,
                                   kernel_initializer = layers.xavier_initializer(),
                                   kernel_regularizer = layers.l2_regularizer(weight_decay),
                                   activation = None)
            if normalizer_fc:
                data = tf.layers.batch_normalization(data, trainable=is_train)
            if activation_fn:
                data = activation_fn(data)
                data = tf.nn.dropout(data, dropout_rate)
    return data


def self_attention(inputs, hidden_state, size, scope_name = "attention", weight_decay = 0.0005):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        attention_context_vec = tf.get_variable(name = "context_vector",
                                        shape= [size],
                                        regularizer = layers.l2_regularizer(scale=weight_decay),
                                        dtype = tf.float32)
        input_projection = layers.fully_connected(inputs, size, activation_fn=tf.nn.tanh,
                                                  weights_regularizer = layers.l2_regularizer(scale=weight_decay))
        vector_atten = tf.reduce_sum(tf.multiply(input_projection, attention_context_vec), axis=2)
        attention_weight  = tf.nn.softmax(vector_atten)
        weights_inputs = tf.multiply(inputs, tf.expand_dims(attention_weight, axis=-1))
        output = tf.reduce_sum(weights_inputs, axis=1)
    return output


def self_attention_last(inputs, hidden_state, size, scope_name = "attention", weight_decay = 0.0005):
    '''
    reference xu et al. 2018
    Unpaired sentiment-to-sentiment translate: a cycled reforcement learning approach
    '''
    assert size == hidden_state.shape[-1]
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        input_projection = tf.contrib.layers.fully_connected(
            inputs, size,
            activation_fn=tf.tanh,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))

        vector_atten = tf.reduce_sum(tf.einsum("aik,ak->aik", input_projection, hidden_state), axis=2)
        attention_weight = tf.nn.softmax(vector_atten)
        weights_inputs = tf.multiply(inputs, tf.expand_dims(attention_weight, axis=-1))
        output = tf.reduce_sum(weights_inputs, axis=1)
    return output


class MultiLossLayer():
    def __init__(self, loss_list):
        self._loss_list = loss_list
        self._sigmas_sq = []
        for i in range(len(self._loss_list)):
            self._sigmas_sq.append(slim.variable('Sigma_sq_' + str(i), dtype=tf.float32, shape=[], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))

    def get_loss(self, eplison = 1e-12):
        factor = tf.div(1.0, tf.multiply(2.0, tf.pow(self._sigmas_sq[0], 2)) + eplison)
        loss = tf.add(tf.multiply(factor, self._loss_list[0]), tf.log(self._sigmas_sq[0] + eplison))
        for i in range(1, len(self._sigmas_sq)):
            factor = tf.div(1.0, tf.multiply(2.0, tf.pow(self._sigmas_sq[i], 2)) + eplison)
            loss = tf.add(loss, tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i] + eplison)))
        return loss


if __name__ == "__main__":
    import numpy as np
    inputs = tf.constant(np.random.rand(32, 8, 16, 16), dtype=tf.float32)
    #print(mobile_net(inputs, num_output=32, weight_decay=0.0005))
    print(hierarchical_1d_CNN(inputs, filter_sizes=(2, 3, 4, 5),num_filters=8, filter_sizes_hierarchical=(2, 3, 4, 5), num_filters_hierarchical=8, dropout_rate=0.8))





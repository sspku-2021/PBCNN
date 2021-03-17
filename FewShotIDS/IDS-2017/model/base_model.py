# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

import tensorflow as tf
from model.utils.modules import MultiLossLayer

from abc import  abstractmethod


class BaseModel():
    def __init__(self, session_length, height, width, num_labels, leaning_rate,
                 num_layers = 1, hidden_size = 300, last_hidden_size = 300,
                 is_train = True, dropout_rate = 0.9, max_grad_norm = 5,
                 l2_regularizer = 0.005,
                 is_pretrain = False, is_time = False, is_size = False,
                 uncertainty = False, is_length = False, early_stop = True, is_tuning = False):

        self.session_length = session_length
        self.height = height
        self.width = width

        self.learning_rate = leaning_rate
        self.num_labels = num_labels

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.last_hidden_size = last_hidden_size

        self.is_train = is_train
        self.dropout_rate = dropout_rate
        # self.dropout_rate =tf.Variable(dropout_rate, trainable=False, dtype=tf.float32)
        self.max_grad_norm = max_grad_norm

        self.l2_regularizer = l2_regularizer

        self.is_pretrain = is_pretrain
        self.is_time = is_time
        self.is_size = is_size
        self.uncertainty = uncertainty
        self.is_length = is_length
        self.early_stop = early_stop
        self.is_tuning = is_tuning

    def placeholder(self):
        self.session = tf.placeholder(tf.float32, shape=[None, self.session_length, self.height, self.width], name="session")
        if self.is_pretrain:
            self.pretrain_placeholder()
        else:
            self.label_placeholder()
        if self.is_length:
            self.length_placeholder()

    def label_placeholder(self):
        self.label = tf.placeholder(tf.int32, shape=[None], name="label")

    def pretrain_placeholder(self):
        if self.is_time:
            self.time = tf.placeholder(tf.float32, shape=[None], name="time")
        if self.is_size:
            self.size = tf.placeholder(tf.float32, shape=[None], name="size")

    def length_placeholder(self):
        self.length = tf.placeholder(tf.int32, [None], name="length")

    def pretrain(self, inputs):
        with tf.variable_scope("pretrain"):
            size_loss, time_loss = None, None
            if self.is_size:
                size_loss, self.size_predict = self.square_loss(inputs, labels=self.size, scope="mtl_size")
            if self.is_time:
                time_loss, self.time_predict = self.square_loss(inputs, labels=self.time, scope="mtl_time")
            if self.is_size and self.is_time:
                if self.uncertainty:
                    multiloss_layer = MultiLossLayer([size_loss, time_loss])
                    square_loss = multiloss_layer.get_loss()
                    self._square_loss = tf.reduce_mean(square_loss, name="square_loss")
                else:
                    square_loss = size_loss + time_loss
                    self._square_loss = tf.reduce_mean(square_loss, name="square_loss")
            elif self.is_size:
                self._square_loss = tf.reduce_mean(size_loss, name="square_loss")
            elif self.is_time:
                self._square_loss = tf.reduce_mean(time_loss, name="square_loss")

            self.cal_cost(self._square_loss)

    def square_loss(self, inputs, labels, scope):
        with tf.variable_scope(scope):
            fc = tf.layers.dense(inputs=inputs,
                                 units=self.last_hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer),
                                 activation=tf.nn.relu)
            outputs = tf.layers.dense(inputs=fc,
                                      units=1,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            loss = tf.square(outputs - labels)
            return loss, outputs

    def category_loss(self, inputs, labels, scope):
        with tf.variable_scope(scope):
            fc = tf.layers.dense(inputs=inputs,
                                 units=self.last_hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer),
                                 activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=fc,
                                      units=self.num_labels,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels), name="loss")
            predict = tf.argmax(tf.nn.softmax(logits), axis=-1, name="predict")
            return loss, predict

    def cal_cost(self, loss):
        with tf.name_scope("cost"):
            self._cost = tf.add(loss, tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
            self._cost_non_reg = loss

    # def cal_acc(self, predicts):
    #     with tf.name_scope("acc"):
    #         _, self._acc = tf.metrics.accuracy(labels=self.label, predictions=predicts)

    def get_train_ops(self, loss, is_tuning = False):
        with tf.variable_scope("train_ops") as scope:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # if is_tuning:
            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            # else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            gradients = [output[0] for output in grads_and_vars]
            variables = [output[1] for output in grads_and_vars]
            gradients = tf.clip_by_global_norm(gradients, clip_norm=self.max_grad_norm)[0]
            self._grad_norm = tf.global_norm(gradients)
            self._train_ops = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    @property
    def cost(self):
        return self._cost

    @property
    def cost_non_reg(self):
        return self._cost_non_reg

    @property
    def predict_category(self):
        return self._category_predicts

    @property
    def train_ops(self):
        return self._train_ops

    # def assign_kp(self, session, kp_value):
    #     session.run(tf.assign(self.dropout_rate, kp_value))
    #
    # def assign_is_train(self, session, is_train):
    #     session.run(tf.assign(self.is_train, is_train))

    def gen_feed_dict(self, batcher):
        feed_dict = {
            self.session: batcher.session,
        }
        if not self.is_pretrain:
            feed_dict.update(
                {self.label: batcher.label}
            )
        if self.is_length:
            feed_dict.update(
                {self.length: batcher.session_length}
            )
        if self.is_time:
            feed_dict.update(
                {self.time: batcher.duration_time}
            )
        if self.is_size:
            feed_dict.update(
                {self.size: batcher.session_size}
            )
        return feed_dict






